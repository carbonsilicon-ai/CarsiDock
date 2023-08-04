from typing import Sequence, Optional, Tuple

import torch
import torch.nn as nn
import sys

from src.layers.openfold.model_utils import OuterProductMean, PairTransition, DropoutRowwise, \
    TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming, TriangleAttention
from src.layers.openfold.tensor_utils import add


class TriAtten(nn.Module):
    def __init__(
            self,
            c_m: int = 512,
            c_z: int = 64,
            c_hidden_opm: int = 64,
            c_hidden_mul: int = 64,
            c_hidden_pair_att: int = 64,
            no_heads_pair: int = 8,
            transition_n: int = 2,
            pair_dropout: float = 0.1,
            inf: float = 1e9,
            eps: float = 1e-6,
            _is_extra_msa_stack: bool = False,
    ):
        super(TriAtten, self).__init__()

        self.outer_product_mean = OuterProductMean(
            c_m,
            c_z,
            c_hidden_opm,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z,
            c_hidden_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z,
            c_hidden_mul,
        )

        self.tri_att_start = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )

        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)

    def forward(self,
                input_tensors: Sequence[torch.Tensor],
                msa_mask: torch.Tensor,
                pair_mask: torch.Tensor,
                chunk_size: Optional[int] = None,
                use_lma: bool = False,
                inplace_safe: bool = False,
                _mask_trans: bool = True,
                _attn_chunk_size: Optional[int] = None,
                _offload_inference: bool = False,
                opm=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # DeepMind doesn't mask these transitions in the source, so _mask_trans
        # should be disabled to better approximate the exact activations of
        # the original.
        pair_trans_mask = pair_mask if _mask_trans else None

        if (_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        m, z = input_tensors

        if opm is None:
            opm = self.outer_product_mean(
                m, mask=msa_mask, chunk_size=chunk_size, inplace_safe=inplace_safe
            )

        z = add(z, opm, inplace=inplace_safe)
        del opm

        tmu_update = self.tri_mul_out(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        z = z + self.ps_dropout_row_layer(tmu_update)

        del tmu_update

        tmu_update = self.tri_mul_in(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        z = z + self.ps_dropout_row_layer(tmu_update)

        del tmu_update

        z = add(z,
                self.ps_dropout_row_layer(
                    self.tri_att_start(
                        z,
                        mask=pair_mask,
                        chunk_size=_attn_chunk_size,
                        use_memory_efficient_kernel=False,
                        use_lma=use_lma,
                        inplace_safe=inplace_safe,
                    )
                ),
                inplace=inplace_safe,
                )

        z = z.transpose(-2, -3)


        z = add(z,
                self.ps_dropout_row_layer(
                    self.tri_att_end(
                        z,
                        mask=pair_mask.transpose(-1, -2),
                        chunk_size=_attn_chunk_size,
                        use_memory_efficient_kernel=False,
                        use_lma=use_lma,
                        inplace_safe=inplace_safe,
                    )
                ),
                inplace=inplace_safe,
                )

        z = z.transpose(-2, -3)

        z = add(z, self.pair_transition(z, mask=pair_trans_mask, chunk_size=chunk_size, ), inplace=inplace_safe, )

        return m, z


if __name__ == '__main__':
    model = TriAtten(512, 64, 64, 64, 64, 4, 2, 0.1, 1e9, 1e-6)
    m = torch.rand(4, 1, 6, 512, device='cuda')
    z = torch.rand(4, 6, 6, 64, device='cuda')
    msa_mask = torch.ones(4, 1, 6, device='cuda')
    msa_mask[0, 0, -2:] = 0
    mask = msa_mask * msa_mask.squeeze().unsqueeze(-1)
    model = model.cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    for _ in range(10000):
        _m, _z = model((m, z), msa_mask, mask)
