import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class BlackLittermanLoss(nn.Module):
    """
    Custom loss function that encourages finding better performing strategies than equal weight
    """
    def __init__(self, returns_df: pd.DataFrame, lam: float = 0.1):
        super(BlackLittermanLoss, self).__init__()
        self.lam = lam
        
        # Calculate equal weight benchmark for reference
        self.num_assets = len(returns_df.columns)
        self.equal_weights = torch.ones(self.num_assets) / self.num_assets

    def forward(self, predictions, targets):
        batch_size = predictions.size(0) // self.num_assets
        
        # Calculate prediction MSE
        mse_loss = F.mse_loss(predictions, targets)
        
        # Get portfolio weights through softmax
        pred_weights = F.softmax(predictions.view(batch_size, self.num_assets), dim=1)
        
        # Calculate portfolio returns
        pred_returns = torch.sum(pred_weights * targets.view(batch_size, self.num_assets), dim=1)
        equal_returns = torch.sum(self.equal_weights.to(predictions.device) * 
                                targets.view(batch_size, self.num_assets), dim=1)
        
        # Calculate return difference from equal weight
        return_diff = pred_returns - equal_returns
        
        # Penalize when returns are worse than equal weight
        # Reward when returns are better
        strategy_loss = -torch.mean(return_diff)
        
        # Combine losses: minimize MSE and maximize return difference
        loss = mse_loss + self.lam * strategy_loss
        
        return loss 