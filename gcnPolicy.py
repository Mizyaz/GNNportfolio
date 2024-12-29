import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
import numpy as np
import gymnasium as gym

from torch_geometric.nn import GCNConv, global_mean_pool

class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that processes graph data using a Graph Neural Network.
    """
    def __init__(self, observation_space: gym.spaces.Dict, **kwargs):
        # Initialize the base class
        features_dim = kwargs.get('features_dim', 128)
        super(GNNFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Extract dimensions from observation space
        num_node_features = observation_space['x'].shape[1]
        num_edge_features = observation_space['edge_attr'].shape[1]
        
        hidden_dims = kwargs.get('hidden_dims', 128)
        hidden_dims2 = kwargs.get('hidden_dims2', 128)

        # Define GCN layers
        self.conv1 = GCNConv(num_node_features, hidden_dims)
        self.conv2 = GCNConv(hidden_dims, hidden_dims2)
        
        # Optionally, process edge attributes if needed
        # For simplicity, we're ignoring edge_attr in this example

        self.pool = global_mean_pool  # Global mean pooling

        # Final linear layer to get the desired features_dim
        self.fc = nn.Linear(hidden_dims2, features_dim)

    def forward(self, observations):
        # observations is a dict with keys 'x', 'edge_index', 'edge_attr'
        x = observations['x']  # [batch_size, num_nodes, num_node_features]
        edge_index = torch.tensor(observations['edge_index'], dtype=torch.int64)  # [2, num_edges]
        edge_index = torch.tensor(observations['edge_index'], dtype=torch.int64)  # [2, num_edges]
        edge_attr = observations['edge_attr']  # [num_edges, num_edge_features]
        # Handle batching properly
        batch_size = x.size(0)
        num_nodes = x.size(1)
        
        # Reshape x to [batch_size * num_nodes, num_features]
        x = x.view(-1, x.size(-1))
        
        # Create proper batch indices for pooling
        batch_idx = torch.arange(batch_size, device=x.device).repeat_interleave(num_nodes)
        
        # Adjust edge indices for batched processing
        edge_index_offset = torch.arange(batch_size, device=edge_index.device) * num_nodes
        edge_index_offset = edge_index_offset.view(-1, 1, 1).expand(-1, 2, edge_index.size(-1))
        edge_index = edge_index + edge_index_offset
        edge_index = edge_index.view(2, -1)
        
        # Handle batching properly
        batch_size = x.size(0)
        num_nodes = x.size(1)
        
        # Reshape x to [batch_size * num_nodes, num_features]
        x = x.view(-1, x.size(-1))
        
        # Create proper batch indices for pooling
        batch_idx = torch.arange(batch_size, device=x.device).repeat_interleave(num_nodes)
        
        # Adjust edge indices for batched processing
        edge_index_offset = torch.arange(batch_size, device=edge_index.device) * num_nodes
        edge_index_offset = edge_index_offset.view(-1, 1, 1).expand(-1, 2, edge_index.size(-1))
        edge_index = edge_index + edge_index_offset
        edge_index = edge_index.view(2, -1)
        
        # Apply GCN layers
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        # Global pooling with proper batch indices
        x = self.pool(x, batch_idx)  # [batch_size, 128]

        # Final linear layer
        x = self.fc(x)  # [batch_size, features_dim]
        
        return x