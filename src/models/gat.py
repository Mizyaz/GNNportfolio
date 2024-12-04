import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import List

class EnhancedPortfolioGAT(nn.Module):
    """
    Enhanced Graph Attention Network for Portfolio Prediction.
    Incorporates batch normalization, skip connections, and a more sophisticated architecture.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 heads: List[int] = [8, 8], dropout: float = 0.6):
        super(EnhancedPortfolioGAT, self).__init__()
        
        # Batch Normalization for input features
        self.bn1 = nn.BatchNorm1d(in_channels)
        
        # First GAT layer
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads[0], dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_channels * heads[0])
        
        # Second GAT layer
        self.conv2 = GATConv(hidden_channels * heads[0], hidden_channels, heads=heads[1], dropout=dropout)
        self.bn3 = nn.BatchNorm1d(hidden_channels * heads[1])
        
        # Prediction head
        self.pred = nn.Sequential(
            nn.Linear(hidden_channels * heads[1], hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        # Skip connection
        self.skip = nn.Linear(in_channels, hidden_channels * heads[1])
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Save input for skip connection
        x_in = x
        
        # Main path
        x = self.bn1(x)
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn2(x)
        x = F.elu(self.conv2(x, edge_index))
        x = self.bn3(x)
        
        # Skip connection
        x = x + self.skip(x_in)
        
        # Prediction
        x = self.pred(x)
        return x 