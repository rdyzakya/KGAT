from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.dense import Linear
import torch
from .base_model import BaseModel

class GATv2Sequential(torch.nn.Sequential):
    def forward(self, x, edge_index, edge_attr, return_attention_weights=None):
        if not return_attention_weights:
            return_attention_weights = None
        all_adj = tuple()
        all_alpha = tuple()
        for module in self._modules.values():
            if return_attention_weights:
                x, (adj, alpha) = module(x, edge_index, edge_attr, return_attention_weights=return_attention_weights)
                all_adj = all_adj + (adj,)
                all_alpha = all_alpha + (alpha,)
            else:
                x = module(x, edge_index, edge_attr, return_attention_weights=return_attention_weights)
        
        if return_attention_weights:
            return x, (all_adj, all_alpha)
        return x

class GATv2Encoder(BaseModel): # using this, because models.GAT don't provide forward method using attention weights
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 num_layers, 
                 heads, 
                 out_channels=None, 
                 negative_slope=0.2, 
                 dropout=0.0, 
                 add_self_loops=True, 
                 bias=True, 
                 share_weights=False,
                 fill_value="mean",
                 **kwargs):
        out_channels = out_channels or in_channels
        super().__init__(in_channels=in_channels, 
                 hidden_channels=hidden_channels, 
                 num_layers=num_layers, 
                 heads=heads, 
                 out_channels=out_channels, 
                 negative_slope=negative_slope, 
                 dropout=dropout, 
                 add_self_loops=add_self_loops, 
                 bias=bias, 
                 share_weights=share_weights,
                 fill_value=fill_value,
                 **kwargs)

        module = [GATv2Conv(in_channels, 
                    hidden_channels, 
                    heads, 
                    concat=True,
                    edge_dim=in_channels, 
                    negative_slope=negative_slope,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                    fill_value=self.fill_value,
                    bias=bias,
                    share_weights=share_weights,
                    **kwargs)] + \
        [GATv2Conv(hidden_channels*heads, 
                    hidden_channels, 
                    heads, 
                    concat=True,
                    edge_dim=in_channels, 
                    negative_slope=negative_slope,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                    fill_value=self.fill_value,
                    bias=bias,
                    share_weights=share_weights,
                    **kwargs) for _ in range(num_layers-2)] + \
        [GATv2Conv(hidden_channels*heads, 
                    out_channels,
                    heads, 
                    concat=False, # output layer out channels, concat = False
                    edge_dim=in_channels, 
                    negative_slope=negative_slope,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                    fill_value=self.fill_value,
                    bias=bias,
                    share_weights=share_weights,
                    **kwargs)] if num_layers > 1 else [GATv2Conv(in_channels, 
                    out_channels,
                    heads, 
                    concat=False, # output layer out channels, concat = False
                    edge_dim=in_channels, 
                    negative_slope=negative_slope,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                    fill_value=self.fill_value,
                    bias=bias,
                    share_weights=share_weights,
                    **kwargs)]
        
        self.gnn = GATv2Sequential(*module)

    def forward(self, x, edge_index, relations, return_attention_weights=None):
        edge_attr = relations[edge_index[1]]
        return self.gnn(x, edge_index[[0,2]], edge_attr, return_attention_weights=return_attention_weights)

class InnerOuterProductDecoder(BaseModel):
    def __init__(self, num_features):
        super().__init__(num_features=num_features)
        self.outer_weight = torch.nn.parameter.Parameter(torch.randn(num_features))
    
    def forward(self, x, edge_index, relations, sigmoid=False):
        """
        R = torch.stack([
            el.outer(self.outer_weight) for el in relations
        ])

        x_R = x[edge_index[0]].matmul(R)

        x_R_x = x_R * x[edge_index[2]]
        x_R_x = x_R_x.sum(dim=2)
        x_R_x = x_R_x[edge_index[1], torch.arange(0,edge_index.shape[1])]

        atau

        x_R = x[edge_index[0]].matmul(R[edge_index[1]])

        x_R_x = x_R * x[edge_index[2]]
        x_R_x = x_R_x.sum(dim=2)
        x_R_x = x_R_x.diagonal()
        """
        out_all = self.forward_all(x, relations, sigmoid)
        return out_all[edge_index[1], edge_index[0], edge_index[2]] # n_edge
    
    def forward_all(self, x, relations, sigmoid=False):
        R = torch.stack([
            el.outer(self.outer_weight) for el in relations
        ])
        
        adj = torch.matmul(
            torch.matmul(x, R),
            x.transpose(0,1)
        ) # n_relation * n_node * n_node

        return torch.sigmoid(adj) if sigmoid else adj

class NodeClassifierDecoder(BaseModel):
    def __init__(self, num_features):
        super().__init__(num_features=num_features)
        # catch multi nuance importance, not only 1 domain, let say a graph with multi domain, this will not only consider 1 domain as the imoprtant one
        self.cls = Linear(in_channels=num_features, out_channels=num_features, bias=False, weight_initializer="glorot")
    
    def forward(self, x, sigmoid=False):
        x = self.cls(x)
        x = x.sum(dim=1)
        x = x.unsqueeze(-1)
        return torch.sigmoid(x) if sigmoid else x