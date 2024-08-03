from .lm import AutoModelForLMKBC
from .injector import Injector, Detach
from .gae import GATv2Encoder, InnerOuterProductDecoder, NodeClassifierDecoder, TripleRetrieval, generate_edge_index
from .aggr import AttentionalAggregation, SoftmaxAggregation
from .base_model import BaseModel
from .graph_prefix import GraphPrefix

from torch_geometric.nn.dense import Linear
from torch.nn import Sequential, ReLU
import torch

class MultiheadGAE(BaseModel):
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
                 subgraph=False,
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
                 subgraph=subgraph,
                 **kwargs)
        
        self.injector = Injector(edge_dim=in_channels)

        self.encoder = GATv2Encoder(
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

        self.detach = Detach()

        if num_layers > 1:
            relation_mlp = [Linear(in_channels=in_channels, out_channels=hidden_channels, bias=True, weight_initializer="glorot"),
                            ReLU()]
            for _ in range(num_layers-2):
                relation_mlp.append(Linear(in_channels=hidden_channels, out_channels=hidden_channels, bias=True, weight_initializer="glorot"))
                relation_mlp.append(ReLU())

            relation_mlp.append(Linear(in_channels=hidden_channels, out_channels=out_channels, bias=True, weight_initializer="glorot"))
            self.relation_mlp = Sequential(*relation_mlp)
        else:
            self.relation_mlp = Linear(in_channels=in_channels, out_channels=out_channels, bias=True, weight_initializer="glorot")

        # self.link_decoder = InnerOuterProductDecoder(num_features=self.encoder.out_channels)
        # self.link_decoder = TripleRetrieval(num_features=self.encoder.out_channels) if subgraph else \
        #     Sequential(Linear(in_channels=self.encoder.out_channels * 3, out_channels=self.encoder.out_channels, bias=False, weight_initializer="glorot"),
        #                ReLU(),
        #                Linear(in_channels=self.encoder.out_channels, out_channels=1, bias=False, weight_initializer="glorot"))
        self.link_decoder = Sequential(Linear(in_channels=self.encoder.out_channels * 3, out_channels=self.encoder.out_channels, bias=False, weight_initializer="glorot"),
                       ReLU(),
                       Linear(in_channels=self.encoder.out_channels, out_channels=1, bias=False, weight_initializer="glorot"))

        # self.node_decoder = NodeClassifierDecoder() if subgraph else Linear(in_channels=self.encoder.out_channels, out_channels=1, bias=False, weight_initializer="glorot")

    def forward(self, 
                x, 
                edge_index, 
                relations, 
                injection_node=None, 
                node_batch=None, 
                injection_node_batch=None, 
                return_attention_weights=None, 
                all=False, 
                sigmoid=False):
        
        if self.subgraph:
            (x, 
             edge_index, 
             relations, 
             x_is_injected, 
             edge_is_injected, 
             relations_is_injected) = self.injector(x, 
                                                     edge_index, 
                                                     relations, 
                                                     injection_node, 
                                                     node_batch=node_batch, 
                                                     injection_node_batch=injection_node_batch)
           
        x = self.encoder(x, edge_index, relations, return_attention_weights=return_attention_weights)
        
        if return_attention_weights:
            x, (all_adj, all_alpha) = x
        else:
            all_adj = None
            all_alpha = None
        
        if self.subgraph:
            (x, edge_index, relations,
             injection_node, _, injection_relation) = self.detach(x, 
                                                                edge_index, 
                                                                relations, 
                                                                x_is_injected, 
                                                                edge_is_injected, 
                                                                relations_is_injected)
        
        relations = self.relation_mlp(relations)
                
        # if all:
        #     out_link = self.link_decoder.forward_all(x, relations, sigmoid=sigmoid)
        # else:
        #     out_link = self.link_decoder.forward(x, edge_index, relations, sigmoid=sigmoid)
        # if self.subgraph:
        #     if all:
        #         out_link = self.link_decoder.forward_all(x,
        #                                                 relations, 
        #                                                 injection_node, 
        #                                                 node_batch=node_batch, 
        #                                                 injection_node_batch=injection_node_batch, 
        #                                                 sigmoid=sigmoid)
        #     else:
        #         out_link = self.link_decoder.forward(x, 
        #                                             edge_index, 
        #                                             relations, 
        #                                             injection_node, 
        #                                             node_batch=node_batch, 
        #                                             injection_node_batch=injection_node_batch, 
        #                                             sigmoid=sigmoid, 
        #                                             allow_intersection=False)
        # else:
        #     link_edge_index = edge_index if not all else generate_edge_index(x.shape[0], relations.shape[0])
        #     inputs = torch.hstack([
        #         relations[link_edge_index[1]],
        #         x[link_edge_index[0]],
        #         x[link_edge_index[2]]
        #     ])

        #     out_link = self.link_decoder(inputs)
        link_edge_index = edge_index if not all else generate_edge_index(x.shape[0], relations.shape[0])
        inputs = torch.hstack([
            relations[link_edge_index[1]],
            x[link_edge_index[0]],
            x[link_edge_index[2]]
        ])

        out_link = self.link_decoder(inputs)
        
        # if self.subgraph:
        #     out_node = self.node_decoder(x, 
        #                                 injection_node, 
        #                                 node_batch=node_batch, 
        #                                 injection_node_batch=injection_node_batch, 
        #                                 sigmoid=sigmoid)
        # else:
        #     out_node = self.node_decoder(x)
        #     out_node = out_node.sigmoid() if sigmoid else out_node

        return x, all_adj, all_alpha, out_link #, out_node