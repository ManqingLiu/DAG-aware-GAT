"""
Standard Graph Attention Network (GAT) Baseline

This module implements a standard GAT that does NOT use causal DAG structure.
Instead, it learns attention weights over a fully-connected graph, serving as
a baseline to compare against the DAG-aware transformer.

Reference: Veličković et al., "Graph Attention Networks", ICLR 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class GATLayer(nn.Module):
    """
    Single Graph Attention Layer
    
    Implements multi-head graph attention without any structural constraints.
    All nodes can attend to all other nodes (fully-connected graph).
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        concat: bool = True,
        leaky_relu_slope: float = 0.2
    ):
        """
        Args:
            in_features: Input feature dimension per node
            out_features: Output feature dimension per head
            num_heads: Number of attention heads
            dropout: Dropout rate for attention coefficients
            concat: If True, concatenate head outputs; else average
            leaky_relu_slope: Negative slope for LeakyReLU
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        # Linear transformation for each head
        self.W = nn.Parameter(torch.zeros(num_heads, in_features, out_features))
        
        # Attention parameters: a = [a_src || a_dst]
        self.a_src = nn.Parameter(torch.zeros(num_heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, out_features, 1))
        
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = nn.Dropout(dropout)
        
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: Node features [batch_size, num_nodes, in_features]
            return_attention: If True, also return attention weights
            
        Returns:
            Updated node features [batch_size, num_nodes, out_features * num_heads] if concat
            else [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Linear transformation: [batch, nodes, heads, out_features]
        # x: [B, N, F_in] -> [B, N, 1, F_in] @ W: [H, F_in, F_out] -> [B, N, H, F_out]
        h = torch.einsum('bni,hio->bnho', x, self.W)
        
        # Compute attention scores
        # Source scores: [B, N, H, 1]
        attn_src = torch.einsum('bnho,hoi->bnhi', h, self.a_src)
        # Destination scores: [B, N, H, 1]
        attn_dst = torch.einsum('bnho,hoi->bnhi', h, self.a_dst)
        
        # Broadcast and sum for pairwise attention: [B, N, N, H]
        # attn_src: [B, N, H, 1] -> [B, N, 1, H]
        # attn_dst: [B, N, H, 1] -> [B, 1, N, H]
        attn_src = attn_src.squeeze(-1).unsqueeze(2)  # [B, N, 1, H]
        attn_dst = attn_dst.squeeze(-1).unsqueeze(1)  # [B, 1, N, H]
        
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        attention = self.leaky_relu(attn_src + attn_dst)  # [B, N, N, H]
        
        # Softmax over source nodes (dim=1)
        attention = F.softmax(attention, dim=1)  # [B, N, N, H]
        attention = self.dropout(attention)
        
        # Aggregate: weighted sum of transformed features
        # attention: [B, N, N, H], h: [B, N, H, F_out]
        # Result: [B, N, H, F_out]
        h_prime = torch.einsum('bmnl,bmlo->bnlo', attention, h)
        
        if self.concat:
            # Concatenate heads: [B, N, H * F_out]
            out = h_prime.reshape(batch_size, num_nodes, -1)
        else:
            # Average heads: [B, N, F_out]
            out = h_prime.mean(dim=2)
        
        if return_attention:
            return out, attention
        return out


class GAT(nn.Module):
    """
    Standard Graph Attention Network (without DAG structure)
    
    This baseline model uses learned attention over a fully-connected graph,
    without incorporating any causal structure from the DAG.
    """
    
    def __init__(
        self,
        dag: Dict,
        network_width: int,
        embedding_dim: int,
        feedforward_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        input_layer_depth: int,
        encoder_weight: float,
        activation: str = "relu",
        use_layernorm: bool = False,
        name: str = None
    ):
        """
        Args:
            dag: DAG dictionary (only used for node structure, not for attention)
            network_width: Width of the MLP layers
            embedding_dim: Dimension of node embeddings
            feedforward_dim: Dimension of feedforward layers (unused, kept for API compatibility)
            num_heads: Number of attention heads
            num_layers: Number of GAT layers
            dropout_rate: Dropout rate
            input_layer_depth: Depth of input MLP
            encoder_weight: Weight for combining GAT output with raw features
            activation: Activation function
            use_layernorm: Whether to use layer normalization
            name: Model name
        """
        super().__init__()
        
        # Store DAG info (for node structure only, NOT for attention mask)
        self.input_nodes = dag['input_nodes']
        self.output_nodes = dag['output_nodes']
        self.node_ids = dag['node_ids']
        self.id2node = {v: k for k, v in self.node_ids.items()}
        
        self.num_nodes = len(self.node_ids.keys())
        self.network_width = network_width
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.encoder_weight = encoder_weight
        self.activation = activation
        self.use_layernorm = use_layernorm
        self.name = name or "gat_baseline"
        
        # Input embedding layer
        self.input_embedding = nn.Linear(1, embedding_dim)
        
        # GAT layers (fully-connected, no DAG constraint)
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else embedding_dim * num_heads
            out_dim = embedding_dim
            concat = True  # Always concat for intermediate layers
            
            self.gat_layers.append(
                GATLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    concat=concat
                )
            )
            
            if use_layernorm:
                self.gat_layers.append(nn.LayerNorm(out_dim * num_heads if concat else out_dim))
            
            self.gat_layers.append(nn.ReLU() if activation == "relu" else nn.GELU())
            self.gat_layers.append(nn.Dropout(dropout_rate))
        
        # Output dimension from GAT
        gat_out_dim = embedding_dim * num_heads
        
        # Projection from GAT output to scalar
        self.embed_to_scalar = nn.Linear(gat_out_dim * (self.num_nodes - 1), 1)
        self.embed_to_scalar_t = nn.Linear(gat_out_dim * (self.num_nodes - 2), 1)
        
        # MLP for y prediction
        self.layer_list = nn.ModuleList()
        for i in range(input_layer_depth):
            if i == 0:
                self.layer_list.append(nn.Linear(self.num_nodes, network_width))
            else:
                self.layer_list.append(nn.Linear(network_width, network_width))
            self.layer_list.append(nn.ReLU())
            self.layer_list.append(nn.Dropout(dropout_rate))
        self.layer_list.append(nn.Linear(network_width, 1))
        
        # MLP for t prediction (propensity)
        self.layer_list_t = nn.ModuleList()
        for i in range(input_layer_depth):
            if i == 0:
                self.layer_list_t.append(nn.Linear(self.num_nodes - 1, network_width))
            else:
                self.layer_list_t.append(nn.Linear(network_width, network_width))
            self.layer_list_t.append(nn.ReLU())
            self.layer_list_t.append(nn.Dropout(dropout_rate))
        self.layer_list_t.append(nn.Linear(network_width, 1))
    
    def forward(self, x: Dict, mask: bool = False, estimator: str = "aipw"):
        """
        Forward pass through GAT.
        
        Note: The 'mask' parameter is ignored - GAT always uses fully-connected attention.
        This is kept for API compatibility with DAGTransformer.
        
        Args:
            x: Dictionary of node values {node_name: tensor}
            mask: Ignored (kept for API compatibility)
            estimator: Type of estimator ("g-formula", "ipw", or "aipw")
            
        Returns:
            Dictionary of outputs based on estimator type
        """
        # Combine all inputs into a single tensor
        combined_input = torch.stack(
            [x[node].float() for node in self.input_nodes.keys()], dim=1
        ).squeeze(-1)
        
        # Create node embeddings
        node_embeddings = []
        for node in self.input_nodes.keys():
            node_input = x[node].float().unsqueeze(-1)  # [B] -> [B, 1]
            node_embedding = self.input_embedding(node_input)  # [B, 1] -> [B, embedding_dim]
            node_embeddings.append(node_embedding)
        
        # Stack: [B, num_nodes, embedding_dim]
        gat_input = torch.stack(node_embeddings, dim=1)
        
        # Process through GAT layers (NO attention mask - fully connected)
        gat_output = gat_input
        for layer in self.gat_layers:
            if isinstance(layer, GATLayer):
                gat_output = layer(gat_output)
            else:
                gat_output = layer(gat_output)
        
        # Generate predictions based on estimator type
        if estimator == "aipw":
            # For t: use all nodes except t and y
            gat_output_t_embeddings = gat_output[:, :-2, :].reshape(gat_output.size(0), -1)
            gat_output_t_scalar = self.embed_to_scalar_t(gat_output_t_embeddings)
            combined_input_t = torch.cat(
                [combined_input[:, :-2], gat_output_t_scalar * self.encoder_weight],
                dim=1
            )
            
            for layer in self.layer_list_t[:-1]:
                combined_input_t = layer(combined_input_t)
            node_output_t = torch.sigmoid(self.layer_list_t[-1](combined_input_t))
            
            # For y: use all nodes except y
            gat_output_y_embeddings = gat_output[:, :-1, :].reshape(gat_output.size(0), -1)
            gat_output_y_scalar = self.embed_to_scalar(gat_output_y_embeddings)
            combined_input_y = torch.cat(
                [combined_input[:, :-1], gat_output_y_scalar * self.encoder_weight],
                dim=1
            )
            
            for layer in self.layer_list[:-1]:
                combined_input_y = layer(combined_input_y)
            node_output_y = self.layer_list[-1](combined_input_y)
            
            node_output = {'y': node_output_y, 't': node_output_t}
            
        elif "ipw" in estimator:
            # For IPW: predict t only
            gat_output_t_embeddings = gat_output[:, :-1, :].reshape(gat_output.size(0), -1)
            gat_output_t_scalar = self.embed_to_scalar(gat_output_t_embeddings)
            combined_input_t = torch.cat(
                [combined_input[:, :-1], gat_output_t_scalar * self.encoder_weight],
                dim=1
            )
            
            for layer in self.layer_list[:-1]:
                combined_input_t = layer(combined_input_t)
            node_output_t = torch.sigmoid(self.layer_list[-1](combined_input_t))
            
            node_output = {'t': node_output_t}
            
        else:  # g-formula
            # For g-formula: predict y only
            gat_output_embeddings = gat_output[:, :-1, :].reshape(gat_output.size(0), -1)
            gat_output_scalar = self.embed_to_scalar(gat_output_embeddings)
            combined_input = torch.cat(
                [combined_input[:, :-1], gat_output_scalar * self.encoder_weight],
                dim=1
            )
            
            for layer in self.layer_list[:-1]:
                combined_input = layer(combined_input)
            node_output_y = self.layer_list[-1](combined_input)
            
            node_output = {'y': node_output_y}
        
        # Build output dictionary
        node_outputs = {}
        for node_name in self.output_nodes.keys():
            node_outputs[node_name] = node_output[node_name]
        
        return node_outputs

