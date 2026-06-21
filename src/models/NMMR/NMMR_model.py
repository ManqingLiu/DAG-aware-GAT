import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import math
from src.models.pure_dag_transformer import CustomTransformerEncoderLayer


class DAGTransformer(nn.Module):
    def __init__(self,
                 dag: Dict,
                 network_width: int,
                 embedding_dim: int,
                 feedforward_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout_rate: float,
                 input_layer_depth: int,
                 encoder_weight: float,
                 activation: str,
                 use_layernorm: bool = False,
                 name: str = None):

        super(DAGTransformer, self).__init__()
        self.input_nodes = dag['input_nodes']
        self.output_nodes = dag['output_nodes']
        self.edges = dag['edges']
        self.node_ids = dag['node_ids']
        self.id2node = {v: k for k, v in self.node_ids.items()}

        self.num_nodes = len(self.node_ids.keys())
        self.network_width = network_width
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.encoder_weight = encoder_weight
        self.activation = activation
        self.use_layernorm = use_layernorm
        self.name = name

        self.adj_matrix = torch.zeros(self.num_nodes, self.num_nodes)

        for source_node_name in self.edges.keys():
            source_node_id = self.node_ids[source_node_name]
            for target_node in self.edges[source_node_name]:
                target_node_id = self.node_ids[target_node]
                self.adj_matrix[source_node_id, target_node_id] = 1

        self.attn_mask = ~(self.adj_matrix.bool().T)


        # Calculate input dimension
        self.input_dim = len(self.input_nodes)

        # Create layer list similar to MLP
        self.layer_list = nn.ModuleList()
        for i in range(input_layer_depth):
            if i == 0:
                self.layer_list.append(nn.Linear(3, self.network_width))
            else:
                self.layer_list.append(nn.Linear(self.network_width, self.network_width))
            self.layer_list.append(nn.ReLU())
            self.layer_list.append(nn.Dropout(self.dropout_rate))

        # Add final layer with output dimension 1
        self.layer_list.append(nn.Linear(self.network_width, 1))

        # Input embedding layer
        self.input_embedding = nn.Linear(1, embedding_dim)

        # Create encoder layers - use custom layer if layernorm is disabled
        if self.use_layernorm:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=self.num_heads,
                dim_feedforward=self.feedforward_dim,
                dropout=self.dropout_rate,
                activation=self.activation,
                batch_first=True
            )
        else:
            encoder_layer = CustomTransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=self.num_heads,
                dim_feedforward=self.feedforward_dim,
                dropout=self.dropout_rate,
                activation=self.activation,
                batch_first=True
            )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Add a linear transformation for a_w_embeddings
        self.embed_to_scalar = nn.Linear(embedding_dim * 2, 1)


    def forward(self, x, mask=False):

        # Combine all inputs into a single tensor
        combined_input = torch.stack([x[node].float() for node in self.input_nodes.keys()], dim=1).squeeze(-1)

        # Transformer part
        node_embeddings = [self.input_embedding(x[node].float()) for node in self.input_nodes.keys()]
        transformer_input = torch.stack(node_embeddings, dim=1)

        # Process the encoder
        if mask==True:
            attn_mask = self.attn_mask.repeat(transformer_input.size(0) * self.num_heads, 1, 1)
            attn_mask = attn_mask.to(transformer_input.device)
            transformer_output = self.encoder(transformer_input, mask=attn_mask)
        else:
            transformer_output = self.encoder(transformer_input)

        # Extract A and W embeddings from transformer output
        a_w_embeddings = transformer_output[:,:2,:].view(transformer_output.size(0), -1)

        a_w_scalar = self.embed_to_scalar(a_w_embeddings)

        # Combine original input with transformer output
        combined_input = torch.cat([combined_input[:, :2], a_w_scalar * self.encoder_weight], dim=1)

        # Process through layers
        for layer in self.layer_list[:-1]:  # All layers except the last
            combined_input = layer(combined_input)

        # Last layer without activation
        node_output = self.layer_list[-1](combined_input)

        # Split the output into individual node outputs
        node_outputs = {}
        for node_name in self.output_nodes.keys():
            node_outputs[node_name] = node_output


        return node_outputs


class MLP_for_NMMR(nn.Module):
    
    def __init__(self, input_dim, train_params):
        super(MLP_for_NMMR, self).__init__()

        self.train_params = train_params
        self.network_width = train_params["network_width"]
        self.network_depth = train_params["network_depth"]

        self.layer_list = nn.ModuleList()
        for i in range(self.network_depth):
            if i == 0:
                self.layer_list.append(nn.Linear(input_dim, self.network_width))
            else:
                self.layer_list.append(nn.Linear(self.network_width, self.network_width))
        self.layer_list.append(nn.Linear(self.network_width, 1))

    def forward(self, x):
        for ix, layer in enumerate(self.layer_list):
            if ix == (self.network_depth + 1):  # if last layer, don't apply relu activation
                x = layer(x)
            else:
                x = torch.relu(layer(x))

        return x

