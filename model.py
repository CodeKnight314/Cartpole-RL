import torch 
import torch.nn as nn 
import torch.nn.functional as F

class CartModel(nn.Module): 
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))