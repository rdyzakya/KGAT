from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import Sequential
from torch.nn import ReLU
from .base_model import BaseModel

class GraphPrefix(BaseModel): # n layer dense (relu in the middle)
    def __init__(self, 
                 num_features, 
                 hidden_channels=None,
                 num_layers=1,
                 n_tokens=1):
        hidden_channels = hidden_channels or num_features
        super().__init__(num_features=num_features, 
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         n_tokens=n_tokens)
        if num_layers == 1:
            self.nn = Linear(in_channels=num_features, 
                          out_channels=num_features*n_tokens, 
                          bias=True, 
                          weight_initializer="glorot")
        else:
            layers = []
            for _ in range(num_layers-1):
                layers.extend([
                    (Linear(in_channels=num_features, 
                          out_channels=num_features, 
                          bias=True, 
                          weight_initializer="glorot"), 'x -> x'),
                    ReLU(inplace=True)
                ])
            
            layers.append(
                (Linear(in_channels=num_features, 
                    out_channels=num_features*n_tokens, 
                    bias=True, 
                    weight_initializer="glorot"), 'x -> x')
            )
            self.nn = Sequential('x', layers)
    
    def forward(self, x):
        out = self.nn(x)
        out = out.view(x.shape[0], self.n_tokens, out.shape[1])
        return out