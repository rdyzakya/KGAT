from .lm import AutoModelForLMKBC
from .injector import Injector, Detach
from .gae import GATv2Encoder, Retrieval, generate_edge_index, VariationalEncoder
from .aggr import AttentionalAggregation, SoftmaxAggregation
from .base_model import BaseModel
from .graph_prefix import GraphPrefix

from torch_geometric.nn.dense import Linear
from torch_geometric.nn.models import VGAE, GAE, MLP
from torch.nn import Sequential, ReLU
import torch

# class MLP(BaseModel):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
#         super().__init__(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers)
        # if num_layers > 1:
        #     nn = [Linear(in_channels=in_channels, out_channels=hidden_channels, bias=True, weight_initializer="glorot"),
        #                     ReLU()]
        #     for _ in range(num_layers-2):
        #         nn.append(Linear(in_channels=hidden_channels, out_channels=hidden_channels, bias=True, weight_initializer="glorot"))
        #         nn.append(ReLU())

        #     nn.append(Linear(in_channels=hidden_channels, out_channels=out_channels, bias=True, weight_initializer="glorot"))
        #     self.nn = Sequential(*nn)
        # else:
        #     self.nn = Linear(in_channels=in_channels, out_channels=out_channels, bias=True, weight_initializer="glorot")
    
#     def forward(self, x):
#         return self.nn(x)

# class ResidualReLUBlock(BaseModel):
#     def __init__(self, in_channels, out_channels, bias=True, weight_initializer="glorot"):
#         super().__init__(in_channels=in_channels, out_channels=out_channels, bias=bias, weight_initializer=weight_initializer)
#         self.lin = Linear(in_channels=in_channels, out_channels=out_channels, bias=bias, weight_initializer=weight_initializer)
#         self.relu = ReLU()
    
#     def forward(self, x):
#         z = self.lin(x)
#         z = self.relu(z)
#         return z + x

# class ResMLP(BaseModel):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
#         super().__init__(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers)
#         if num_layers > 1:
#             nn = [ResidualReLUBlock(in_channels=in_channels, out_channels=hidden_channels, bias=True, weight_initializer="glorot")]
#             for _ in range(num_layers-2):
#                 nn.append(ResidualReLUBlock(in_channels=hidden_channels, out_channels=hidden_channels, bias=True, weight_initializer="glorot"))

#             nn.append(ResidualReLUBlock(in_channels=hidden_channels, out_channels=out_channels, bias=True, weight_initializer="glorot"))
#             self.nn = Sequential(*nn)
#         else:
#             self.nn = ResidualReLUBlock(in_channels=in_channels, out_channels=out_channels, bias=True, weight_initializer="glorot")
    
#     def forward(self, x):
#         return self.nn(x)


class MyModel(BaseModel):
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

        # self.encoder = GATv2Encoder(
        #     in_channels=in_channels,
        #     hidden_channels=hidden_channels,
        #     num_layers=num_layers,
        #     heads=heads,
        #     out_channels=out_channels,
        #     negative_slope=negative_slope,
        #     dropout=dropout,
        #     add_self_loops=add_self_loops,
        #     bias=bias,
        #     share_weights=share_weights,
        #     **kwargs
        # )

        encoder = GATv2Encoder(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            num_layers=num_layers, 
            heads=heads, 
            out_channels=out_channels, 
            negative_slope=negative_slope, 
            dropout=dropout, 
            add_self_loops=add_self_loops, 
            bias=bias, 
            share_weights=share_weights,
            **kwargs
        )

        decoder = Retrieval()

        self.gae = GAE(encoder=encoder, decoder=decoder)

        self.f_r1 = Linear(in_channels=self.gae.encoder.out_channels*3, 
                                 out_channels=self.gae.encoder.out_channels, 
                                 bias=False, 
                                 weight_initializer="glorot") # f reference
        self.f_r2 = MLP(in_channels=self.gae.encoder.out_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers)
        
        self.g_q = Linear(in_channels=self.gae.encoder.out_channels, 
                                 out_channels=self.gae.encoder.out_channels, 
                                 bias=True, 
                                 weight_initializer="glorot") # g query
        
        # self.retrieval = Retrieval()
    
    # def encode(self, x, edge_index, relations, return_attention_weights=None):
    #     x = self.encoder(x, edge_index, relations, return_attention_weights=return_attention_weights)
        
    #     if return_attention_weights:
    #         x, (all_adj, all_alpha) = x
    #     else:
    #         all_adj = None
    #         all_alpha = None
        
    #     return x, all_adj, all_alpha
    
    def triple_emb(self, x, edge_index, relations):
        inputs = torch.hstack([
            relations[edge_index[1]],
            x[edge_index[0]],
            x[edge_index[2]]
        ])
        emb = self.f_r1(inputs)
        return self.f_r2(emb) + emb
        
    def forward(self, 
                x, # reference 
                edge_index, 
                relations, 
                query,
                values=None,
                node_batch=None, 
                query_batch=None,
                values_batch=None,
                sigmoid=False,
                allow_intersection=False):
        # encode reference
        z = self.gae.encode(x, edge_index, relations)
        
        # prepare reference triples
        reference = self.triple_emb(z, edge_index, relations)

        node_batch = node_batch if node_batch is not None else torch.zeros(x.shape[0]).int()

        source_batch = node_batch[edge_index[0]]
        tgt_batch = node_batch[edge_index[2]]
        
        if (source_batch != tgt_batch).any() and not allow_intersection:
            raise ValueError(f"Intersection between batch, there are connection between different graph \n source_batch : {source_batch} \n tgt_batch : {tgt_batch}")
        
        reference_batch = source_batch

        # prepare queries
        query = self.g_q(query)
        query_batch = query_batch if query_batch is not None else torch.zeros(query.shape[0]).int()

        # prepare values
        values_batch = values_batch if values_batch is not None else torch.zeros(values.shape[0]).int()

        # out
        query_reference_out = self.gae.decode(reference, query, value_batch=reference_batch, query_batch=query_batch, sigmoid=sigmoid)

        out = (query_reference_out,)

        if values is not None:
            if len(values) > 0:
                reference_values_out = self.gae.decode(reference, values, value_batch=reference_batch, query_batch=values_batch, sigmoid=sigmoid)
                query_values_out = self.gae.decode(query, values, value_batch=query_batch, query_batch=values_batch, sigmoid=sigmoid)

                out += (reference_values_out, query_values_out)


        return out[0] if len(out) == 1 else out