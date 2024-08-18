from torch_geometric.nn.dense.linear import Linear
from .base_model import BaseModel
import torch

class GraphPrefix(BaseModel):
    def __init__(self, in_channels, d_model, n_token, bias=True):
        super().__init__(in_channels=in_channels, d_model=d_model, n_token=n_token, bias=bias)
        self.lin = Linear(in_channels=in_channels, out_channels=d_model*n_token, bias=bias, weight_initializer="glorot")
    
    def forward(self, x):
        return self.lin(x)
    
    @staticmethod
    def load(path):
        dump = torch.load(path)
        model = GraphPrefix(**dump["attribute"])
        model.load_state_dict(dump["state_dict"])
        return model