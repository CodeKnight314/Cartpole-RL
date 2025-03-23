import torch 
import torch.nn as nn 
import torch.nn.functional as F
import os 

class CartModel(nn.Module): 
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        
        self.fc = nn.Sequential(*[
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ])

    def forward(self, x):
        return self.fc(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
        else: 
            raise FileNotFoundError(f"File not found: {path}")