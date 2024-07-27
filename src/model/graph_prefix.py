import torch
from torch_geometric.nn.dense.linear import Linear
from .base_model import BaseModel

class GraphPrefix(BaseModel):
    def __init__(self, 
                 num_features, 
                 n_tokens=1):
        super().__init__(num_features=num_features, 
                         n_tokens=n_tokens)
        self.lin = Linear(in_channels=num_features, 
                          out_channels=num_features*n_tokens, 
                          bias=True, 
                          weight_initializer="glorot")
    
    def forward(self, x):
        out = self.lin(x)
        out = out.view(x.shape[0], self.n_tokens, out.shape[1])
        return out