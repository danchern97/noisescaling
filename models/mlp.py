import torch
import torch.nn as nn

__all__ = ["MLP"]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, activation=nn.ReLU):
        super(MLP, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
            
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
