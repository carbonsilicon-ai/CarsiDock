import torch as th
import torch.nn.functional as F
import dgl
import numpy as np
import random
import dgl.function as fn
from torch import nn
import pandas as pd

##
#the model architecture of graph transformer is modified from https://github.com/BioinfoMachineLearning/DeepInteract

def glorot_orthogonal(tensor, scale):
	"""Initialize a tensor's values according to an orthogonal Glorot initialization scheme."""
	if tensor is not None:
		th.nn.init.orthogonal_(tensor.data)
		scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
		tensor.data *= scale.sqrt()


class MultiHeadAttentionLayer(nn.Module):
	"""Compute attention scores with a DGLGraph's node and edge (geometric) features."""
	def __init__(self, num_input_feats, num_output_feats,
				num_heads, using_bias=False, update_edge_feats=True):
		super(MultiHeadAttentionLayer, self).__init__()
		
        # Declare shared variables
		self.num_output_feats = num_output_feats
		self.num_heads = num_heads
		self.using_bias = using_bias
		self.update_edge_feats = update_edge_feats
		
		# Define node features' query, key, and value tensors, and define edge features' projection tensors
		self.Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
		self.K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
		self.V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
		self.edge_feats_projection = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
		
		self.reset_parameters()
		
	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		scale = 2.0
		if self.using_bias:
			glorot_orthogonal(self.Q.weight, scale=scale)
			self.Q.bias.data.fill_(0)
			
			glorot_orthogonal(self.K.weight, scale=scale)
			self.K.bias.data.fill_(0)
			
			glorot_orthogonal(self.V.weight, scale=scale)
			self.V.bias.data.fill_(0)
			
			glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
			self.edge_feats_projection.bias.data.fill_(0)
		else:
			glorot_orthogonal(self.Q.weight, scale=scale)
			glorot_orthogonal(self.K.weight, scale=scale)
			glorot_orthogonal(self.V.weight, scale=scale)
			glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
	
	def propagate_attention(self, g):
		# Compute attention scores
		g.apply_edges(lambda edges: {"score": edges.src['K_h'] * edges.dst['Q_h']})
		# Scale and clip attention scores
		g.apply_edges(lambda edges: {"score": (edges.data["score"]/np.sqrt(self.num_output_feats)).clamp(-5.0,5.0)})		
		# Use available edge features to modify the attention scores
		g.apply_edges(lambda edges: {"score": edges.data['score'] * edges.data['proj_e']})
		# Copy edge features as e_out to be passed to edge_feats_MLP
		if self.update_edge_feats:
			g.apply_edges(lambda edges: {"e_out": edges.data["score"]})
		
		# Apply softmax to attention scores, followed by clipping
		g.apply_edges(lambda edges: {"score": th.exp((edges.data["score"].sum(-1, keepdim=True)).clamp(-5.0,5.0))})
		# Send weighted values to target nodes
		#e_ids = g.edges()
		#g.send_and_recv(e_ids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
		#g.send_and_recv(e_ids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))
		g.update_all(fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
		g.update_all(fn.copy_e('score', 'score'), fn.sum('score', 'z'))
	
	def forward(self, g, node_feats, edge_feats):
		with g.local_scope():
			e_out = None
			node_feats_q = self.Q(node_feats)
			node_feats_k = self.K(node_feats)
			node_feats_v = self.V(node_feats)
			edge_feats_projection = self.edge_feats_projection(edge_feats)			
			# Reshape tensors into [num_nodes, num_heads, feat_dim] to get projections for multi-head attention
			g.ndata['Q_h'] = node_feats_q.view(-1, self.num_heads, self.num_output_feats)
			g.ndata['K_h'] = node_feats_k.view(-1, self.num_heads, self.num_output_feats)
			g.ndata['V_h'] = node_feats_v.view(-1, self.num_heads, self.num_output_feats)
			g.edata['proj_e'] = edge_feats_projection.view(-1, self.num_heads, self.num_output_feats)
			# Disperse attention information
			self.propagate_attention(g)
			# Compute final node and edge representations after multi-head attention
			h_out = g.ndata['wV'] / (g.ndata['z'] + th.full_like(g.ndata['z'], 1e-6))  # Add eps to all
			if self.update_edge_feats:
				e_out = g.edata['e_out']
		# Return attention-updated node and edge representations
		return h_out, e_out


class GraphTransformerModule(nn.Module):
	"""A Graph Transformer module (equivalent to one layer of graph convolutions)."""
	def __init__(
			self,
			num_hidden_channels,
			activ_fn=nn.SiLU(),
			residual=True,
			num_attention_heads=4,
			norm_to_apply='batch',
			dropout_rate=0.1,
			num_layers=4,
			):
		super(GraphTransformerModule, self).__init__()
		
		# Record parameters given
		self.activ_fn = activ_fn
		self.residual = residual
		self.num_attention_heads = num_attention_heads
		self.norm_to_apply = norm_to_apply
		self.dropout_rate = dropout_rate
		self.num_layers = num_layers
		
		self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()
		
		self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
		if self.apply_layer_norm:
			self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
			self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
		else:  # Otherwise, default to using batch normalization
			self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
			self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)
		
		self.mha_module = MultiHeadAttentionLayer(
			self.num_hidden_channels,
			self.num_output_feats // self.num_attention_heads,
			self.num_attention_heads,
			self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
			update_edge_feats=True
		)
		
		self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
		self.O_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
		
		# MLP for node features
		dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
		self.node_feats_MLP = nn.ModuleList([
			nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
			self.activ_fn,
			dropout,
			nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
		])
		
		if self.apply_layer_norm:
			self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
			self.layer_norm2_edge_feats = nn.LayerNorm(self.num_output_feats)
		else:  # Otherwise, default to using batch normalization
			self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
			self.batch_norm2_edge_feats = nn.BatchNorm1d(self.num_output_feats)
		
		# MLP for edge features
		self.edge_feats_MLP = nn.ModuleList([
			nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
			self.activ_fn,
			dropout,
			nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
		])
		
		self.reset_parameters()
	
	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		scale = 2.0
		glorot_orthogonal(self.O_node_feats.weight, scale=scale)
		self.O_node_feats.bias.data.fill_(0)
		glorot_orthogonal(self.O_edge_feats.weight, scale=scale)
		self.O_edge_feats.bias.data.fill_(0)
		
		for layer in self.node_feats_MLP:
			if hasattr(layer, 'weight'):  # Skip initialization for activation functions
				glorot_orthogonal(layer.weight, scale=scale)
		
		for layer in self.edge_feats_MLP:
			if hasattr(layer, 'weight'):
				glorot_orthogonal(layer.weight, scale=scale)
	
	def run_gt_layer(self, g, node_feats, edge_feats):
		"""Perform a forward pass of graph attention using a multi-head attention (MHA) module."""
		node_feats_in1 = node_feats  # Cache node representations for first residual connection
		edge_feats_in1 = edge_feats  # Cache edge representations for first residual connection
			
		# Apply first round of normalization before applying graph attention, for performance enhancement
		if self.apply_layer_norm:
			node_feats = self.layer_norm1_node_feats(node_feats)
			edge_feats = self.layer_norm1_edge_feats(edge_feats)
		else:  # Otherwise, default to using batch normalization
			node_feats = self.batch_norm1_node_feats(node_feats)
			edge_feats = self.batch_norm1_edge_feats(edge_feats)
		
		# Get multi-head attention output using provided node and edge representations
		node_attn_out, edge_attn_out = self.mha_module(g, node_feats, edge_feats)
		
		node_feats = node_attn_out.view(-1, self.num_output_feats)
		edge_feats = edge_attn_out.view(-1, self.num_output_feats)
		
		node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
		edge_feats = F.dropout(edge_feats, self.dropout_rate, training=self.training)
		
		node_feats = self.O_node_feats(node_feats)
		edge_feats = self.O_edge_feats(edge_feats)
		
		# Make first residual connection
		if self.residual:
			node_feats = node_feats_in1 + node_feats  # Make first node residual connection
			edge_feats = edge_feats_in1 + edge_feats  # Make first edge residual connection
		
		node_feats_in2 = node_feats  # Cache node representations for second residual connection
		edge_feats_in2 = edge_feats  # Cache edge representations for second residual connection
		
		# Apply second round of normalization after first residual connection has been made
		if self.apply_layer_norm:
			node_feats = self.layer_norm2_node_feats(node_feats)
			edge_feats = self.layer_norm2_edge_feats(edge_feats)
		else:  # Otherwise, default to using batch normalization
			node_feats = self.batch_norm2_node_feats(node_feats)
			edge_feats = self.batch_norm2_edge_feats(edge_feats)
		
		# Apply MLPs for node and edge features
		for layer in self.node_feats_MLP:
			node_feats = layer(node_feats)
		for layer in self.edge_feats_MLP:
			edge_feats = layer(edge_feats)
		
		# Make second residual connection
		if self.residual:
			node_feats = node_feats_in2 + node_feats  # Make second node residual connection
			edge_feats = edge_feats_in2 + edge_feats  # Make second edge residual connection
		
		# Return edge representations along with node representations (for tasks other than interface prediction)
		return node_feats, edge_feats
	
	def forward(self, g, node_feats, edge_feats):
		"""Perform a forward pass of a Graph Transformer to get intermediate node and edge representations."""
		node_feats, edge_feats = self.run_gt_layer(g, node_feats, edge_feats)
		return node_feats, edge_feats


class FinalGraphTransformerModule(nn.Module):
	"""A (final layer) Graph Transformer module that combines node and edge representations using self-attention."""	
	def __init__(self,
				num_hidden_channels,
				activ_fn=nn.SiLU(),
				residual=True,
				num_attention_heads=4,
				norm_to_apply='batch',
				dropout_rate=0.1,
				num_layers=4):
		super(FinalGraphTransformerModule, self).__init__()
		
		# Record parameters given
		self.activ_fn = activ_fn
		self.residual = residual
		self.num_attention_heads = num_attention_heads
		self.norm_to_apply = norm_to_apply
		self.dropout_rate = dropout_rate
		self.num_layers = num_layers
		self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()
		
		self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
		if self.apply_layer_norm:
			self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
			self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
		else:  # Otherwise, default to using batch normalization
			self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
			self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)
		
		self.mha_module = MultiHeadAttentionLayer(
					self.num_hidden_channels,
					self.num_output_feats // self.num_attention_heads,
					self.num_attention_heads,
					self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
					update_edge_feats=False)
		
		self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
		
		# MLP for node features
		dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
		self.node_feats_MLP = nn.ModuleList([
					nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
					self.activ_fn,
					dropout,
					nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
					])
		
		if self.apply_layer_norm:
			self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
		else:  # Otherwise, default to using batch normalization
			self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
		
		self.reset_parameters()
	
	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		scale = 2.0
		glorot_orthogonal(self.O_node_feats.weight, scale=scale)
		self.O_node_feats.bias.data.fill_(0)
		
		for layer in self.node_feats_MLP:
			if hasattr(layer, 'weight'):  # Skip initialization for activation functions
				glorot_orthogonal(layer.weight, scale=scale)
		
		#glorot_orthogonal(self.conformation_module.weight, scale=scale)
	
	def run_gt_layer(self, g, node_feats, edge_feats):
		"""Perform a forward pass of graph attention using a multi-head attention (MHA) module."""
		node_feats_in1 = node_feats  # Cache node representations for first residual connection
		
		# Apply first round of normalization before applying graph attention, for performance enhancement
		if self.apply_layer_norm:
			node_feats = self.layer_norm1_node_feats(node_feats)
			edge_feats = self.layer_norm1_edge_feats(edge_feats)
		else:  # Otherwise, default to using batch normalization
			node_feats = self.batch_norm1_node_feats(node_feats)
			edge_feats = self.batch_norm1_edge_feats(edge_feats)
		
		# Get multi-head attention output using provided node and edge representations
		node_attn_out, _ = self.mha_module(g, node_feats, edge_feats)
		node_feats = node_attn_out.view(-1, self.num_output_feats)		
		node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
		node_feats = self.O_node_feats(node_feats)
		
		# Make first residual connection
		if self.residual:
			node_feats = node_feats_in1 + node_feats  # Make first node residual connection
		
		node_feats_in2 = node_feats  # Cache node representations for second residual connection
		
		# Apply second round of normalization after first residual connection has been made
		if self.apply_layer_norm:
			node_feats = self.layer_norm2_node_feats(node_feats)
		else:  # Otherwise, default to using batch normalization
			node_feats = self.batch_norm2_node_feats(node_feats)
		
		# Apply MLP for node features
		for layer in self.node_feats_MLP:
			node_feats = layer(node_feats)
		
		# Make second residual connection
		if self.residual:
			node_feats = node_feats_in2 + node_feats  # Make second node residual connection
		
		# Return node representations
		return node_feats
	
	def forward(self, g, node_feats, edge_feats):
		"""Perform a forward pass of a Graph Transformer to get final node representations."""
		node_feats = self.run_gt_layer(g, node_feats, edge_feats)
		return node_feats


class DGLGraphTransformer(nn.Module):
	"""A graph transformer, as a DGL module.
	"""
	def __init__(
			self,
			in_channels, 
			edge_features=10,
			num_hidden_channels=128,
            activ_fn=nn.SiLU(),
			transformer_residual=True,
			num_attention_heads=4,
			norm_to_apply='batch',
			dropout_rate=0.1,
			num_layers=4,
			**kwargs
			):
		"""Graph Transformer Layer
		
		Parameters
		----------
		in_channels : int
			Input channel size for nodes.
		edge_features : int
			Input channel size for edges.			
		num_hidden_channels : int
			Hidden channel size for both nodes and edges.
		activ_fn : Module
			Activation function to apply in MLPs.
		transformer_residual : bool
			Whether to use a transformer-residual update strategy for node features.
		num_attention_heads : int
			How many attention heads to apply to the input node features in parallel.
		norm_to_apply : str
			Which normalization scheme to apply to node and edge representations (i.e. 'batch' or 'layer').
		dropout_rate : float
			How much dropout (i.e. forget rate) to apply before activation functions.
		num_layers : int
			How many layers of geometric attention to apply.
		"""
		super(DGLGraphTransformer, self).__init__()
		
		# Initialize model parameters
		self.activ_fn = activ_fn
		self.transformer_residual = transformer_residual
		self.num_attention_heads = num_attention_heads
		self.norm_to_apply = norm_to_apply
		self.dropout_rate = dropout_rate
		self.num_layers = num_layers
		
		# --------------------
		# Initializer Modules
		# --------------------
		# Define all modules related to edge and node initialization
		self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
		self.edge_encoder = nn.Linear(edge_features, num_hidden_channels) 
        # --------------------
		# Transformer Module
		# --------------------
		# Define all modules related to a variable number of Graph Transformer modules
		num_intermediate_layers = max(0, num_layers - 1)
		gt_block_modules = [GraphTransformerModule(
										num_hidden_channels=num_hidden_channels,
										activ_fn=activ_fn,
										residual=transformer_residual,
										num_attention_heads=num_attention_heads,
										norm_to_apply=norm_to_apply,
										dropout_rate=dropout_rate,
										num_layers=num_layers) for _ in range(num_intermediate_layers)]
		if num_layers > 0:
			gt_block_modules.extend([
							FinalGraphTransformerModule(
										num_hidden_channels=num_hidden_channels,
										activ_fn=activ_fn,
										residual=transformer_residual,
										num_attention_heads=num_attention_heads,
										norm_to_apply=norm_to_apply,
										dropout_rate=dropout_rate,
										num_layers=num_layers)])
		self.gt_block = nn.ModuleList(gt_block_modules)
	
	def forward(self, g, node_feats, edge_feats):		
		node_feats = self.node_encoder(node_feats)
		edge_feats = self.edge_encoder(edge_feats)
		
		g.ndata['x'] = node_feats
		g.edata['h'] = edge_feats	
		# Apply a given number of intermediate graph attention layers to the node and edge features given
		for gt_layer in self.gt_block[:-1]:
			node_feats, edge_feats = gt_layer(g, node_feats, edge_feats)
		
		# Apply final layer to update node representations by merging current node and edge representations
		node_feats = self.gt_block[-1](g, node_feats, edge_feats)
		return node_feats


def to_dense_batch_dgl(bg, feats, fill_value=0):
	max_num_nodes = int(bg.batch_num_nodes().max())
	#batch = feats.new_zeros(feats.size(0), dtype=torch.long)
	#batch = th.cat([th.full((1,int(x.cpu().numpy())), y) for x,y in zip(bg.batch_num_nodes(),range(bg.batch_size))],dim=1).reshape(-1).type(th.long)
	batch = th.cat([th.full((1,x.type(th.int)), y) for x,y in zip(bg.batch_num_nodes(),range(bg.batch_size))],dim=1).reshape(-1).type(th.long).to(bg.device)
	cum_nodes = th.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])
	idx = th.arange(bg.num_nodes(), dtype=th.long, device=bg.device)
	idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
	size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]
	out = feats.new_full(size, fill_value)
	out[idx] = feats
	out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])
	
	mask = th.zeros(bg.batch_size * max_num_nodes, dtype=th.bool,
					device=bg.device)
	mask[idx] = 1
	mask = mask.view(bg.batch_size, max_num_nodes)  
	return out, mask

			
class RTMScore(nn.Module):
	def __init__(self, lig_model, prot_model, in_channels, hidden_dim, n_gaussians, dropout_rate=0.15, 
					dist_threhold=1000):
		super(RTMScore, self).__init__()
		
		self.lig_model = lig_model
		self.prot_model = prot_model
		self.MLP = nn.Sequential(nn.Linear(in_channels*2, hidden_dim), 
								nn.BatchNorm1d(hidden_dim), 
								nn.ELU(), 
								nn.Dropout(p=dropout_rate)
								) 
		self.z_pi = nn.Linear(hidden_dim, n_gaussians)
		self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
		self.z_mu = nn.Linear(hidden_dim, n_gaussians)
		self.atom_types = nn.Linear(in_channels, 17)
		self.bond_types = nn.Linear(in_channels*2, 4)
		
		self.dist_threhold = dist_threhold	
    
	def forward(self, bgp, bgl):		
		h_l = self.lig_model(bgl, bgl.ndata['atom'].float(), bgl.edata['bond'].float())
		h_p = self.prot_model(bgp, bgp.ndata['feats'].float(), bgp.edata['feats'].float())
		
		h_l_x, l_mask = to_dense_batch_dgl(bgl, h_l, fill_value=0)
		h_p_x, p_mask = to_dense_batch_dgl(bgp, h_p, fill_value=0)

		h_l_pos, _ =  to_dense_batch_dgl(bgl, bgl.ndata["pos"], fill_value=0)
		h_p_pos, _ =  to_dense_batch_dgl(bgp, bgp.ndata["pos"], fill_value=0)
				
		(B, N_l, C_out), N_p = h_l_x.size(), h_p_x.size(1)
		self.B = B
		self.N_l = N_l
		self.N_p = N_p
		
		# Combine and mask
		h_l_x = h_l_x.unsqueeze(-2)
		h_l_x = h_l_x.repeat(1, 1, N_p, 1) # [B, N_l, N_t, C_out]
		
		h_p_x = h_p_x.unsqueeze(-3)
		h_p_x = h_p_x.repeat(1, N_l, 1, 1) # [B, N_l, N_t, C_out]
		
		C = th.cat((h_l_x, h_p_x), -1)
		self.C_mask = C_mask = l_mask.view(B, N_l, 1) & p_mask.view(B, 1, N_p)
		self.C = C = C[C_mask]
		C = self.MLP(C)
		
		# Get batch indexes for ligand-target combined features
		C_batch = th.tensor(range(B)).unsqueeze(-1).unsqueeze(-1)
		C_batch = C_batch.repeat(1, N_l, N_p)[C_mask]
			
		# Outputs
		pi = F.softmax(self.z_pi(C), -1)
		sigma = F.elu(self.z_sigma(C))+1.1
		mu = F.elu(self.z_mu(C))+1
		atom_types = self.atom_types(h_l)
		bond_types = self.bond_types(th.cat([h_l[bgl.edges()[0]],h_l[bgl.edges()[1]]], axis=1))
		
		dist = self.compute_euclidean_distances_matrix(h_l_pos, h_p_pos.view(B,-1,3))[C_mask]
		return pi, sigma, mu, dist.unsqueeze(1).detach(), atom_types, bond_types, C_batch
	
	def compute_euclidean_distances_matrix(self, X, Y):
		# Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
		# (X-Y)^2 = X^2 + Y^2 -2XY
		X = X.double()
		Y = Y.double()
		
		dists = -2 * th.bmm(X, Y.permute(0, 2, 1)) + th.sum(Y**2,    axis=-1).unsqueeze(1) + th.sum(X**2, axis=-1).unsqueeze(-1)	
		return th.nan_to_num((dists**0.5).view(self.B, self.N_l,-1,24),10000).min(axis=-1)[0]

