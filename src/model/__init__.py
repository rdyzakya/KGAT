from .lm import AutoModelForLMKBC
from .injector import Injector, Detach
from .gae import GATv2Encoder, InnerOuterProductDecoder, NodeClassifierDecoder
from .aggr import AttentionalAggregation, SoftmaxAggregation
from .base_model import BaseModel
from .graph_prefix import GraphPrefix

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
        
        self.injector = Injector(edge_dim=in_channels) if subgraph else None

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

        self.link_decoder = InnerOuterProductDecoder(num_features=self.encoder.out_channels)

        self.node_decoder = NodeClassifierDecoder(num_features=self.encoder.out_channels)
    
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
            x, edge_index, relations = self.detach(x, 
                                                   edge_index, 
                                                   relations, 
                                                   x_is_injected, 
                                                   edge_is_injected, 
                                                   relations_is_injected)
        
        if all:
            out_link = self.link_decoder.forward_all(x, relations, sigmoid=sigmoid)
        else:
            out_link = self.link_decoder.forward(x, edge_index, relations, sigmoid=sigmoid)
        
        out_node = self.node_decoder(x, sigmoid=sigmoid)

        return x, all_adj, all_alpha, out_link, out_node