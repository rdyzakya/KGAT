from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.aggr import MeanAggregation
import torch
# from torch_geometric.nn.dense import Linear
from .base_model import BaseModel
from itertools import product

class ReLUGATv2(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_channels = kwargs.get("in_channels")
        out_channels = kwargs.get("out_channels")
        heads = kwargs.get("heads")
        bias = kwargs.get("bias")
        self.lin = Linear(in_channels=in_channels, out_channels=out_channels*heads, bias=bias, weight_initializer="glorot")
        self.gatv2 = GATv2Conv(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return_attention_weights = kwargs.get("return_attention_weights")
        x = args[0]
        out = self.gatv2.forward(*args, **kwargs)
        if return_attention_weights:
            z, (adj, alpha) = out
            return z.relu() + self.lin(x), (adj, alpha)
            # return z.relu(), (adj, alpha)
        else:
            z = out
            return z.relu() + self.lin(x)
            # return z.relu()

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
                 **kwargs)
        
        self.self_loop_edge_attr = torch.nn.parameter.Parameter(
            torch.randn(in_channels)
        )

        module = [ReLUGATv2(
                    in_channels=in_channels, 
                    out_channels=hidden_channels ,
                    heads=heads, 
                    concat=True,
                    edge_dim=in_channels, 
                    negative_slope=negative_slope,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                    fill_value=self.self_loop_edge_attr,
                    bias=bias,
                    share_weights=share_weights,
                    **kwargs)] + \
        [ReLUGATv2(
                    in_channels=hidden_channels*heads, 
                    out_channels=hidden_channels, 
                    heads=heads, 
                    concat=True,
                    edge_dim=in_channels, 
                    negative_slope=negative_slope,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                    fill_value=self.self_loop_edge_attr,
                    bias=bias,
                    share_weights=share_weights,
                    **kwargs) for _ in range(num_layers-2)] + \
        [ReLUGATv2(
                    in_channels=hidden_channels*heads, 
                    out_channels=out_channels,
                    heads=heads, 
                    concat=False, # output layer out channels, concat = False
                    edge_dim=in_channels, 
                    negative_slope=negative_slope,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                    fill_value=self.self_loop_edge_attr,
                    bias=bias,
                    share_weights=share_weights,
                    **kwargs)] if num_layers > 1 else [
                    ReLUGATv2(in_channels, 
                    out_channels,
                    heads, 
                    concat=False, # output layer out channels, concat = False
                    edge_dim=in_channels, 
                    negative_slope=negative_slope,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                    fill_value=self.self_loop_edge_attr,
                    bias=bias,
                    share_weights=share_weights,
                    **kwargs)
                    ]
        
        self.gnn = GATv2Sequential(*module)

    def forward(self, x, edge_index, relations, return_attention_weights=None):
        edge_attr = relations[edge_index[1]]
        ei = edge_index[[0,2]]
        out = self.gnn(x, ei, edge_attr, return_attention_weights=return_attention_weights)
        return out

class VariationalEncoder(BaseModel):
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
                 **kwargs)
        self.encoder_mu = GATv2Encoder(in_channels=in_channels, 
                 hidden_channels=hidden_channels, 
                 num_layers=num_layers, 
                 heads=heads, 
                 out_channels=out_channels, 
                 negative_slope=negative_slope, 
                 dropout=dropout, 
                 add_self_loops=add_self_loops, 
                 bias=bias, 
                 share_weights=share_weights,
                 **kwargs)
        self.encoder_std = GATv2Encoder(in_channels=in_channels, 
                 hidden_channels=hidden_channels, 
                 num_layers=num_layers, 
                 heads=heads, 
                 out_channels=out_channels, 
                 negative_slope=negative_slope, 
                 dropout=dropout, 
                 add_self_loops=add_self_loops, 
                 bias=bias, 
                 share_weights=share_weights,
                 **kwargs)
        
    def forward(self, x, edge_index, relations):
        mu = self.encoder_mu(x, edge_index, relations, return_attention_weights=False)
        std = self.encoder_std(x, edge_index, relations, return_attention_weights=False)
        return mu, std

# class InnerOuterProductDecoder(BaseModel):
#     def __init__(self, num_features):
#         super().__init__(num_features=num_features)
#         self.outer_weight = torch.nn.parameter.Parameter(torch.randn(num_features))
    
#     def forward(self, x, edge_index, relations, sigmoid=False):
#         """
#         R = torch.stack([
#             el.outer(self.outer_weight) for el in relations
#         ])

#         x_R = x[edge_index[0]].matmul(R)

#         x_R_x = x_R * x[edge_index[2]]
#         x_R_x = x_R_x.sum(dim=2)
#         x_R_x = x_R_x[edge_index[1], torch.arange(0,edge_index.shape[1])]

#         atau

#         x_R = x[edge_index[0]].matmul(R[edge_index[1]])

#         x_R_x = x_R * x[edge_index[2]]
#         x_R_x = x_R_x.sum(dim=2)
#         x_R_x = x_R_x.diagonal()
#         """
#         out_all = self.forward_all(x, relations, sigmoid)
#         return out_all[edge_index[1], edge_index[0], edge_index[2]] # n_edge
    
#     def forward_all(self, x, relations, sigmoid=False):
#         R = torch.stack([
#             el.outer(self.outer_weight) / self.num_features**0.5 for el in relations # normalization using dimension
#         ])
        
#         adj = torch.matmul(
#             torch.matmul(x, R),
#             x.transpose(0,1)
#         ) # n_relation * n_node * n_node

#         return torch.sigmoid(adj) if sigmoid else adj

class Retrieval(BaseModel):
    def __init__(self):
        super().__init__()
        self.mean = MeanAggregation()

    def forward(self, value, query, value_batch=None, query_batch=None, sigmoid=False):
        value_batch = torch.zeros(value.shape[0]) if value_batch is None else value_batch
        query_batch = torch.arange(0, query.shape[0]) if query_batch is None else query_batch
        result = torch.mm(value, query.t())
        result = self.mean(result.t(), index=query_batch)
        result = result.t()
        result = result[torch.arange(0, result.shape[0]), value_batch]
        result = result.unsqueeze(1)
        return torch.sigmoid(result) if sigmoid else result

def generate_edge_index(n_node, n_relation):
    return torch.tensor(list(product(range(n_node), range(n_relation), range(n_node))))

# class TripleRetrieval(BaseModel):
#     def __init__(self, num_features):
#         super().__init__(num_features=num_features)
#         self.lin_emb = Linear(in_channels=num_features*3, out_channels=num_features, bias=True, weight_initializer="glorot")
#         self.mean = MeanAggregation()
    
#     def forward(self, x, edge_index, relations, injection_node, node_batch=None, injection_node_batch=None, sigmoid=False, allow_intersection=False):
#         node_batch = torch.zeros(x.shape[0]) if node_batch is None else node_batch
#         injection_node_batch = torch.arange(0, injection_node.shape[0]) if injection_node_batch is None else injection_node_batch

        # source_batch = node_batch[edge_index[0]]
        # tgt_batch = node_batch[edge_index[2]]
        
        # if (source_batch != tgt_batch).any() and not allow_intersection:
        #     raise ValueError(f"Intersection between batch, there are connection between different graph \n source_batch : {source_batch} \n tgt_batch : {tgt_batch}")
        
        # triple_batch = source_batch
        
#         triple_emb = self.triple_emb(x, edge_index, relations)

#         return self.retrieve(triple_emb, injection_node, triple_batch, injection_node_batch, sigmoid=sigmoid)
    
#     def forward_all(self, x, relations, injection_node, node_batch=None, injection_node_batch=None, sigmoid=False):
#         edge_index = generate_edge_index(x.shape[0], relations.shape[0])
#         out = self.forward(x, edge_index, relations, injection_node, node_batch, injection_node_batch, sigmoid, allow_intersection=True)
#         adj = out.view(relations.shape[0], x.shape[0], x.shape[0])
#         return adj
    
#     def triple_emb(self, x, edge_index, relations):
#         inputs = torch.hstack([
#             relations[edge_index[1]],
#             x[edge_index[0]],
#             x[edge_index[2]]
#         ])
#         return self.lin_emb(inputs)
    
#     def retrieve(self, triple_emb, injection_node, triple_batch, injection_node_batch, sigmoid=False):
#         result = torch.mm(triple_emb, injection_node.t())
#         result = self.mean(result.t(), index=injection_node_batch)
#         result = result.t()
#         result = result[torch.arange(0, result.shape[0]), triple_batch]
#         result = result.unsqueeze(1)
#         return result.sigmoid() if sigmoid else result