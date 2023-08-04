"""

"""

from typing import Optional
import torch.nn.functional as F
import torch
from torch.distributions import Normal

from src.layers.modeling_outputs import DockingPoseOutput
from src.layers.openfold.modeling_evoformer import TriAtten
from src.layers.modeling_layers import TransformerEncoderLayer
import torch.nn as nn

from src.modeling.modeling_base_model import BaseModel, BetaConfig, BaseForDockingPose, NonLinearHead


class TransformerEncoderWithTriangle(nn.Module):
    def __init__(
            self,
            encoder_layers: int = 6,
            embed_dim: int = 768,
            ffn_embed_dim: int = 3072,
            attention_heads: int = 8,
            emb_dropout: float = 0.1,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.0,
            max_seq_len: int = 256,
            activation_fn: str = "gelu",
            post_ln: bool = False,
            no_final_head_layer_norm: bool = False,
    ) -> None:

        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = nn.LayerNorm(self.embed_dim)
        if not post_ln:
            self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None

        if not no_final_head_layer_norm:
            self.final_head_layer_norm = nn.LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.folds = TriAtten(c_m=self.embed_dim, c_z=attention_heads, c_hidden_opm=attention_heads,
                              c_hidden_mul=attention_heads, c_hidden_pair_att=attention_heads, no_heads_pair=2)
        self.x_ln = nn.LayerNorm(self.embed_dim)
        self.pair_ln = nn.LayerNorm(attention_heads)

    def forward(
            self,
            emb: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        bsz = emb.size(0)
        seq_len = emb.size(1)
        if padding_mask is not None:
            # [bsz, n]
            fold_msa_mask = (~padding_mask).float()
            fold_pair_mask = fold_msa_mask.unsqueeze(1) * fold_msa_mask.unsqueeze(-1)
            fold_msa_mask = fold_msa_mask.unsqueeze(1)
        else:
            fold_msa_mask = torch.ones(bsz, 1, seq_len, dtype=torch.long, device=emb.device)
            fold_pair_mask = torch.ones(bsz, seq_len, seq_len, dtype=torch.long, device=emb.device)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        assert attn_mask is not None

        # attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)

        def transpose_pair(pair_repr, bsz, seq_len, c):
            return pair_repr.view(bsz, c, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()

        def de_transpose_pair(pair_repr, bsz, seq_len, c):
            return pair_repr.permute(0, 3, 1, 2).contiguous().view(bsz * c, seq_len, seq_len)

        m_pair_repr = attn_mask
        for i in range(len(self.layers)):
            x, _, attn_probs = self.layers[i](
                x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
            )
            m_pair_repr += attn_probs
        m_pair_repr[m_pair_repr == float('-inf')] = 0
        m_pair_repr = (
            m_pair_repr.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )

        x, m_pair_repr = self.folds(
            (x.unsqueeze(1), m_pair_repr),
            fold_msa_mask,
            fold_pair_mask,
            # use_lma=True
        )
        x = x.squeeze(1)
        # m_pair_repr = m_pair_repr / 6
        attn_mask = de_transpose_pair(m_pair_repr, bsz, seq_len, self.attention_heads)

        def norm_loss(x, eps=1e-10, tolerance=1.0):
            x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x ** 2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (
                    torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
            ).mean()

        x_norm = norm_loss(x)
        token_mask = 1.0 - input_padding_mask.float()
        x_norm = masked_mean(token_mask, x_norm)

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = norm_loss(delta_pair_repr)
        delta_pair_repr_norm = masked_mean(
            pair_mask, delta_pair_repr_norm, dim=(-1, -2)
        )

        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm


class FoldModel(BaseModel):
    def __init__(self, config: BetaConfig):
        super().__init__(config)
        self.encoder = TransformerEncoderWithTriangle(
            encoder_layers=config.num_hidden_layers,
            embed_dim=config.hidden_size,
            ffn_embed_dim=config.intermediate_size,
            attention_heads=config.num_attention_heads,
            emb_dropout=config.hidden_dropout_prob,
            dropout=config.hidden_dropout_prob,
            attention_dropout=config.attention_probs_dropout_prob,
            activation_dropout=config.activation_dropout,
            max_seq_len=config.max_seq_len,
            activation_fn=config.hidden_act,
            no_final_head_layer_norm=config.delta_pair_repr_norm_loss < 0,
            post_ln=config.post_ln,
        )

class FoldForDockingPose(BaseForDockingPose):
    def __init__(self, config: BetaConfig):
        super(FoldForDockingPose, self).__init__(config)
        self.mol_model = FoldModel(config.mol_config)
        self.pocket_model = FoldModel(config.pocket_config)
        self.concat_encoder = FoldModel(config)

    @staticmethod
    def distance(m, n):
        _m = torch.mean(m, dim=2, keepdim=True)
        _n = torch.mean(n, dim=1, keepdim=True)
        return _m - _n

    def forward(
            self,
            mol_src_tokens,
            mol_src_distance,
            mol_src_edge_type,
            pocket_src_tokens,
            pocket_src_distance,
            pocket_src_edge_type,
            masked_tokens=None,
            distance_target=None,
            holo_distance_target=None,
            dist_threshold=0,
            **kwargs
    ):
        mol_padding_mask = mol_src_tokens.eq(0)
        pocket_padding_mask = pocket_src_tokens.eq(0)

        mol_outputs = self.mol_model(src_tokens=mol_src_tokens, src_distance=mol_src_distance,
                                     src_edge_type=mol_src_edge_type)
        mol_encoder_rep = mol_outputs.last_hidden_state
        mol_encoder_pair_rep = mol_outputs.last_pair_repr

        pocket_outputs = self.pocket_model(src_tokens=pocket_src_tokens, src_distance=pocket_src_distance,
                                           src_edge_type=pocket_src_edge_type)
        pocket_encoder_rep = pocket_outputs.last_hidden_state
        pocket_encoder_pair_rep = pocket_outputs.last_pair_repr

        bsz, mol_sz = mol_encoder_rep.shape[:2]
        pocket_sz = pocket_encoder_rep.size(1)

        concat_rep = torch.cat(
            [mol_encoder_rep, pocket_encoder_rep], dim=-2
        )  # [batch, mol_sz+pocket_sz, hidden_dim]
        concat_mask = torch.cat(
            [mol_padding_mask, pocket_padding_mask], dim=-1
        )  # [batch, mol_sz+pocket_sz]

        concat_pair_rep = torch.zeros(bsz, mol_sz + pocket_sz, mol_sz + pocket_sz, self.config.num_attention_heads,
                                      device=mol_src_tokens.device)
        concat_pair_rep[:, :mol_sz, :mol_sz] += mol_encoder_pair_rep
        concat_pair_rep[:, mol_sz:, mol_sz:] += pocket_encoder_pair_rep
        # concat_pair_rep[:, :mol_sz, mol_sz:] += self.distance(mol_encoder_pair_rep, pocket_encoder_pair_rep)
        # concat_pair_rep[:, mol_sz:, :mol_sz] += self.distance(pocket_encoder_pair_rep, mol_encoder_pair_rep)
        # concat_pair_rep = (concat_pair_rep + concat_pair_rep.transpose(1, 2)) / 2

        decoder_rep = concat_rep
        decoder_pair_rep = concat_pair_rep
        for i in range(self.config.recycling):
            binding_outputs = self.concat_encoder(seq_rep=decoder_rep, pair_rep=decoder_pair_rep,
                                                  padding_mask=concat_mask)
            decoder_rep = binding_outputs.last_hidden_state
            decoder_pair_rep = binding_outputs.last_pair_repr

        mol_decoder = decoder_rep[:, :mol_sz]
        pocket_decoder = decoder_rep[:, mol_sz:]

        mol_pair_decoder_rep = decoder_pair_rep[:, :mol_sz, :mol_sz, :]
        mol_pocket_pair_decoder_rep = (decoder_pair_rep[:, :mol_sz, mol_sz:, :] + decoder_pair_rep[:, mol_sz:, :mol_sz,
                                                                                  :].transpose(1, 2)) / 2.0
        mol_pocket_pair_decoder_rep[mol_pocket_pair_decoder_rep == float("-inf")] = 0

        cross_rep = torch.cat(
            [
                mol_pocket_pair_decoder_rep,
                mol_decoder.unsqueeze(-2).repeat(1, 1, pocket_sz, 1),
                pocket_decoder.unsqueeze(-3).repeat(1, mol_sz, 1, 1),
            ],
            dim=-1,
        )  # [batch, mol_sz, pocket_sz, 4*hidden_size]

        cross_distance_predict = (
                F.elu(self.cross_distance_project(cross_rep).squeeze(-1)) + 1.0
        )  # batch, mol_sz, pocket_sz

        holo_encoder_pair_rep = torch.cat(
            [
                mol_pair_decoder_rep,
                mol_decoder.unsqueeze(-2).repeat(1, 1, mol_sz, 1),
            ],
            dim=-1,
        )  # [batch, mol_sz, mol_sz, 3*hidden_size]
        holo_distance_predict = self.holo_distance_project(holo_encoder_pair_rep)  # batch, mol_sz, mol_sz

        distance_mask = distance_target.ne(0)  # 0 is padding
        if dist_threshold > 0:
            distance_mask &= (distance_target < dist_threshold)
        distance_predict = cross_distance_predict[distance_mask]
        distance_target = distance_target[distance_mask]
        distance_loss = F.mse_loss(distance_predict.float(), distance_target.float(), reduction="mean")

        ### holo distance loss
        holo_distance_mask = holo_distance_target.ne(0)  # 0 is padding
        holo_distance_predict_train = holo_distance_predict[holo_distance_mask]
        holo_distance_target = holo_distance_target[holo_distance_mask]
        holo_distance_loss = F.smooth_l1_loss(
            holo_distance_predict_train.float(),
            holo_distance_target.float(),
            reduction="mean",
            beta=1.0,
        )

        loss = distance_loss + holo_distance_loss
        return DockingPoseOutput(
            loss=loss,
            cross_loss=distance_loss,
            holo_loss=holo_distance_loss,
            cross_distance_predict=cross_distance_predict,
            holo_distance_predict=holo_distance_predict
        )


class RtmScoreHead(nn.Module):
    def __init__(self, hidden_size, n_gaussian):
        super().__init__()
        # self.z_pi = nn.Linear(hidden_size, n_gaussian)
        # self.z_sigma = nn.Linear(hidden_size, n_gaussian)
        # self.z_mu = nn.Linear(hidden_size, n_gaussian)
        self.z_pi = NonLinearHead(hidden_size, n_gaussian, 'relu')
        self.z_sigma = NonLinearHead(hidden_size, n_gaussian, 'relu')
        self.z_mu = NonLinearHead(hidden_size, n_gaussian, 'relu')

    def forward(self, features, dist, dist_mask=None, eps=1e-10, threshold=8):
        if dist_mask is None:
            dist_mask = torch.ones_like(dist, dtype=torch.bool, device=dist.device)
        if threshold > 0:
            dist_mask = dist_mask & (dist <= threshold)

        pi = F.softmax(self.z_pi(features), -1)  #
        sigma = F.elu(self.z_sigma(features)) + 1.4  #
        mu = F.elu(self.z_mu(features)) + 1

        normal = Normal(mu, sigma)
        loglik = normal.log_prob(dist.unsqueeze(-1).expand_as(normal.loc))
        candidate_loss = -torch.logsumexp(torch.log(pi + eps) + loglik, dim=-1)
        # candidate_loss2 = ((mu-dist.unsqueeze(-1)).abs()-sigma)**2
        loss = candidate_loss[dist_mask].mean()

        # logprob = loglik + torch.log(pi)
        # prob = logprob.exp().sum(-1)

        prob = loglik.exp() * pi / (dist ** 2 + eps).unsqueeze(-1)
        prob = prob.sum(-1)
        score = torch.stack([p[m].sum() for p, m in zip(prob, dist_mask)])
        return score, loss, (pi, mu, sigma)

    @staticmethod
    def mdn_loss_fn(pi, sigma, mu, y, eps=1e-10):
        normal = Normal(mu, sigma)
        # loss = th.exp(normal.log_prob(y.expand_as(normal.loc)))
        # loss = th.sum(loss * pi, dim=1)
        # loss = -th.log(loss)
        loglik = normal.log_prob(y.expand_as(normal.loc))
        loss = -torch.logsumexp(torch.log(pi + eps) + loglik, dim=1)
        return loss


class FoldForDocking(BaseForDockingPose):
    def __init__(self, config: BetaConfig):
        super(FoldForDocking, self).__init__(config)
        self.mol_model = FoldModel(config.mol_config)
        self.pocket_model = FoldModel(config.pocket_config)
        self.concat_encoder = FoldModel(config)
        # self.cross_distance_project = nn.Linear(config.hidden_size * 2 + config.num_attention_heads, 1)
        # self.holo_distance_project = nn.Linear(config.hidden_size + config.num_attention_heads, 1)
        self.regression_head2 = NonLinearHead(2 * config.hidden_size + config.num_attention_heads, 3, 'relu')
        # self.regression_head = nn.Linear(2 * config.hidden_size + config.num_attention_heads, 1)
        self.rtm_score_head = RtmScoreHead(config.hidden_size * 2 + config.num_attention_heads, 10)

    @staticmethod
    def distance(m, n):
        _m = torch.mean(m, dim=2, keepdim=True)
        _n = torch.mean(n, dim=1, keepdim=True)
        return _m - _n

    def forward(
            self,
            mol_src_tokens,
            mol_src_distance,
            mol_src_edge_type,
            pocket_src_tokens,
            pocket_src_distance,
            pocket_src_edge_type,
            masked_tokens=None,
            distance_target=None,
            holo_distance_target=None,
            dist_threshold=0,
            score=None,
            **kwargs
    ):
        mol_padding_mask = mol_src_tokens.eq(0)
        pocket_padding_mask = pocket_src_tokens.eq(0)

        mol_outputs = self.mol_model(src_tokens=mol_src_tokens, src_distance=mol_src_distance,
                                     src_edge_type=mol_src_edge_type)
        mol_encoder_rep = mol_outputs.last_hidden_state
        mol_encoder_pair_rep = mol_outputs.last_pair_repr

        pocket_outputs = self.pocket_model(src_tokens=pocket_src_tokens, src_distance=pocket_src_distance,
                                           src_edge_type=pocket_src_edge_type)
        pocket_encoder_rep = pocket_outputs.last_hidden_state
        pocket_encoder_pair_rep = pocket_outputs.last_pair_repr

        bsz, mol_sz = mol_encoder_rep.shape[:2]
        pocket_sz = pocket_encoder_rep.size(1)

        concat_rep = torch.cat(
            [mol_encoder_rep, pocket_encoder_rep], dim=-2
        )  # [batch, mol_sz+pocket_sz, hidden_dim]
        concat_mask = torch.cat(
            [mol_padding_mask, pocket_padding_mask], dim=-1
        )  # [batch, mol_sz+pocket_sz]

        concat_pair_rep = torch.zeros(bsz, mol_sz + pocket_sz, mol_sz + pocket_sz, self.config.num_attention_heads,
                                      device=mol_src_tokens.device)
        concat_pair_rep[:, :mol_sz, :mol_sz] += mol_encoder_pair_rep
        concat_pair_rep[:, mol_sz:, mol_sz:] += pocket_encoder_pair_rep
        concat_pair_rep[:, :mol_sz, mol_sz:] += self.distance(mol_encoder_pair_rep, pocket_encoder_pair_rep)
        concat_pair_rep[:, mol_sz:, :mol_sz] += self.distance(pocket_encoder_pair_rep, mol_encoder_pair_rep)
        # concat_pair_rep = (concat_pair_rep + concat_pair_rep.transpose(1, 2)) / 2

        decoder_rep = concat_rep
        decoder_pair_rep = concat_pair_rep
        for i in range(self.config.recycling):
            binding_outputs = self.concat_encoder(seq_rep=decoder_rep, pair_rep=decoder_pair_rep,
                                                  padding_mask=concat_mask)
            decoder_rep = binding_outputs.last_hidden_state
            decoder_pair_rep = binding_outputs.last_pair_repr

        mol_decoder = decoder_rep[:, :mol_sz]
        pocket_decoder = decoder_rep[:, mol_sz:]

        mol_pair_decoder_rep = decoder_pair_rep[:, :mol_sz, :mol_sz, :]
        mol_pair_decoder_rep = (mol_pair_decoder_rep + mol_pair_decoder_rep.transpose(1, 2)) / 2.0
        mol_pocket_pair_decoder_rep = (decoder_pair_rep[:, :mol_sz, mol_sz:, :] + decoder_pair_rep[:, mol_sz:, :mol_sz,
                                                                                  :].transpose(1, 2)) / 2.0
        mol_pocket_pair_decoder_rep[mol_pocket_pair_decoder_rep == float("-inf")] = 0

        cross_rep = torch.cat(
            [
                mol_pocket_pair_decoder_rep,
                mol_decoder.unsqueeze(-2).repeat(1, 1, pocket_sz, 1),
                pocket_decoder.unsqueeze(-3).repeat(1, mol_sz, 1, 1),
            ],
            dim=-1,
        )  # [batch, mol_sz, pocket_sz, 4*hidden_size]

        cross_distance_predict = (
                F.elu(self.cross_distance_project(cross_rep).squeeze(-1)) + 1.0
        )  # batch, mol_sz, pocket_sz

        holo_encoder_pair_rep = torch.cat(
            [
                mol_pair_decoder_rep,
                mol_decoder.unsqueeze(-2).repeat(1, 1, mol_sz, 1),
            ],
            dim=-1,
        )  # [batch, mol_sz, mol_sz, hidden_size+head_num]
        holo_distance_predict = self.holo_distance_project(holo_encoder_pair_rep).squeeze(-1)  # batch, mol_sz, mol_sz
        holo_distance_predict = F.elu(holo_distance_predict) + 1

        # rtm score loss
        if distance_target is None:
            distance_mask = mol_padding_mask.unsqueeze(-1) & pocket_padding_mask.unsqueeze(-2)
            rtm_score, rtm_loss = self.rtm_score_head(cross_rep, cross_distance_predict, distance_mask)
        else:
            rtm_score, rtm_loss = self.rtm_score_head(cross_rep, distance_target, distance_target.ne(0))

        # distance
        distance_mask = distance_target.ne(0)  # 0 is padding
        if dist_threshold > 0:
            distance_mask &= (distance_target < dist_threshold)
        distance_predict = cross_distance_predict[distance_mask]
        distance_target_train = distance_target[distance_mask]
        distance_loss = F.mse_loss(distance_predict.float(), distance_target_train.float(), reduction="mean")

        ### holo distance loss
        holo_distance_mask = holo_distance_target.ne(0)  # 0 is padding
        holo_distance_predict_train = holo_distance_predict[holo_distance_mask]
        holo_distance_target = holo_distance_target[holo_distance_mask]
        holo_distance_loss = F.smooth_l1_loss(
            holo_distance_predict_train.float(),
            holo_distance_target.float(),
            reduction="mean",
            beta=1.0,
        )

        # affinity loss
        affinity_features = torch.stack([f[m].mean(0) for f, m in zip(cross_rep, distance_target.ne(0))])
        score_indices = score > 0
        affinity_pred = self.regression_head2(affinity_features)[score_indices]
        affinity_loss = F.smooth_l1_loss(affinity_pred, score[score_indices], reduction='mean', beta=1.0)

        loss = distance_loss + holo_distance_loss + affinity_loss + rtm_loss
        return DockingPoseOutput(
            loss=loss,
            score_loss=rtm_loss,
            affinity_loss=affinity_loss,
            cross_loss=distance_loss,
            holo_loss=holo_distance_loss,
            affinity_predict=affinity_pred,
            rtm_score=rtm_score,
            cross_distance_predict=cross_distance_predict,
            holo_distance_predict=holo_distance_predict
        )


class FoldDockingForPredict(FoldForDocking):
    def forward(
            self,
            mol_src_tokens,
            mol_src_distance,
            mol_src_edge_type,
            pocket_src_tokens,
            pocket_src_distance,
            pocket_src_edge_type,
            masked_tokens=None,
            distance_target=None,
            holo_distance_target=None,
            dist_threshold=0,
            score=None,
            **kwargs
    ):
        mol_padding_mask = mol_src_tokens.eq(0)
        pocket_padding_mask = pocket_src_tokens.eq(0)

        if distance_target is None:
            dist_mol_mask = mol_src_tokens > 2
            dist_pocket_mask = pocket_src_tokens > 2
            dist_mask = dist_mol_mask.unsqueeze(-1) & dist_pocket_mask.unsqueeze(-2)
        else:
            dist_mask = distance_target.ne(0)

        mol_outputs = self.mol_model(src_tokens=mol_src_tokens, src_distance=mol_src_distance,
                                     src_edge_type=mol_src_edge_type)
        mol_encoder_rep = mol_outputs.last_hidden_state
        mol_encoder_pair_rep = mol_outputs.last_pair_repr

        pocket_outputs = self.pocket_model(src_tokens=pocket_src_tokens, src_distance=pocket_src_distance,
                                           src_edge_type=pocket_src_edge_type)
        pocket_encoder_rep = pocket_outputs.last_hidden_state
        pocket_encoder_pair_rep = pocket_outputs.last_pair_repr

        bsz, mol_sz = mol_encoder_rep.shape[:2]
        pocket_sz = pocket_encoder_rep.size(1)

        concat_rep = torch.cat(
            [mol_encoder_rep, pocket_encoder_rep], dim=-2
        )  # [batch, mol_sz+pocket_sz, hidden_dim]
        concat_mask = torch.cat(
            [mol_padding_mask, pocket_padding_mask], dim=-1
        )  # [batch, mol_sz+pocket_sz]

        concat_pair_rep = torch.zeros(bsz, mol_sz + pocket_sz, mol_sz + pocket_sz, self.config.num_attention_heads,
                                      device=mol_src_tokens.device)
        concat_pair_rep[:, :mol_sz, :mol_sz] += mol_encoder_pair_rep
        concat_pair_rep[:, mol_sz:, mol_sz:] += pocket_encoder_pair_rep
        concat_pair_rep[:, :mol_sz, mol_sz:] += self.distance(mol_encoder_pair_rep, pocket_encoder_pair_rep)
        concat_pair_rep[:, mol_sz:, :mol_sz] += self.distance(pocket_encoder_pair_rep, mol_encoder_pair_rep)
        # concat_pair_rep = (concat_pair_rep + concat_pair_rep.transpose(1, 2)) / 2

        decoder_rep = concat_rep
        decoder_pair_rep = concat_pair_rep
        for i in range(self.config.recycling):
            binding_outputs = self.concat_encoder(seq_rep=decoder_rep, pair_rep=decoder_pair_rep,
                                                  padding_mask=concat_mask)
            decoder_rep = binding_outputs.last_hidden_state
            decoder_pair_rep = binding_outputs.last_pair_repr

        mol_decoder = decoder_rep[:, :mol_sz]
        pocket_decoder = decoder_rep[:, mol_sz:]

        mol_pair_decoder_rep = decoder_pair_rep[:, :mol_sz, :mol_sz, :]
        mol_pair_decoder_rep = (mol_pair_decoder_rep + mol_pair_decoder_rep.transpose(1, 2)) / 2.0
        mol_pocket_pair_decoder_rep = (decoder_pair_rep[:, :mol_sz, mol_sz:, :] + decoder_pair_rep[:, mol_sz:, :mol_sz,
                                                                                  :].transpose(1, 2)) / 2.0
        mol_pocket_pair_decoder_rep[mol_pocket_pair_decoder_rep == float("-inf")] = 0

        cross_rep = torch.cat(
            [
                mol_pocket_pair_decoder_rep,
                mol_decoder.unsqueeze(-2).repeat(1, 1, pocket_sz, 1),
                pocket_decoder.unsqueeze(-3).repeat(1, mol_sz, 1, 1),
            ],
            dim=-1,
        )  # [batch, mol_sz, pocket_sz, 4*hidden_size]

        cross_distance_predict = (
                F.elu(self.cross_distance_project(cross_rep).squeeze(-1)) + 1.0
        )  # batch, mol_sz, pocket_sz

        holo_encoder_pair_rep = torch.cat(
            [
                mol_pair_decoder_rep,
                mol_decoder.unsqueeze(-2).repeat(1, 1, mol_sz, 1),
            ],
            dim=-1,
        )  # [batch, mol_sz, mol_sz, hidden_size+head_num]
        holo_distance_predict = self.holo_distance_project(holo_encoder_pair_rep).squeeze(-1)  # batch, mol_sz, mol_sz
        holo_distance_predict = F.elu(holo_distance_predict) + 1

        if distance_target is None:
            rtm_score, rtm_loss, mdn = self.rtm_score_head(cross_rep, cross_distance_predict, dist_mask, threshold=8)
        else:
            rtm_score, rtm_loss, mdn = self.rtm_score_head(cross_rep, distance_target, dist_mask, threshold=8)

        # affinity loss
        affinity_features = torch.stack([f[m].mean(0) for f, m in zip(cross_rep, dist_mask)])
        affinity_pred = self.regression_head2(affinity_features)

        return DockingPoseOutput(
            affinity_predict=affinity_pred,
            rtm_score=rtm_score,
            cross_distance_predict=cross_distance_predict,
            holo_distance_predict=holo_distance_predict,
            mdn=mdn
        )

    def load_state_dict(self, state_dict, strict=True, remove_prefix='model.'):
        new_state_dict = dict()
        if len(remove_prefix) > 0:
            for k, v in state_dict.items():
                if k.startswith(remove_prefix):
                    new_state_dict[k[len(remove_prefix):]] = v
                else:
                    new_state_dict[k] = v
        else:
            new_state_dict = state_dict
        super(FoldDockingForPredict, self).load_state_dict(new_state_dict, strict)
