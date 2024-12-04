from torch_geometric.data import Dataset, Data
import torch

class GraphDataset(Dataset):
    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs
        
    def len(self):
        return len(self.graphs)
        
    def get(self, idx):
        return self.graphs[idx]

def create_datasets(train_graphs):
    """Create train and validation datasets."""
    train_size = int(0.8 * len(train_graphs))
    val_graphs = train_graphs[train_size:]
    train_graphs = train_graphs[:train_size]
    
    return GraphDataset(train_graphs), GraphDataset(val_graphs) 