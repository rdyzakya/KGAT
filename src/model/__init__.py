from .lm import AutoModelForLMKBC
from .injector import Injector, Detach
from .gae import KGATEncoder, KGATRetrieval, KGATTripleEmb, KGATModel
from .aggr import AttentionalAggregation, SoftmaxAggregation
from .base_model import BaseModel
from .graph_prefix import GraphPrefix

from torch_geometric.utils import softmax

import torch_geometric.typing
from torch_geometric import is_compiling
from torch_geometric.utils import scatter, segment

def expand_left(ptr, dim, dims):
    for _ in range(dims + dim if dim < 0 else dim):
        ptr = ptr.unsqueeze(0)
    return ptr

class Pipeline(BaseModel):
    def __init__(self, kgat_model, graph_prefix, language_model):
        super().__init__()
        self.kgat_model = kgat_model
        self.graph_prefix = graph_prefix
        self.language_model = language_model
        # self.readout = AttentionalAggregation()
    
    def forward(self, x, edge_index, relations, query, node_batch, query_batch, n_token):

        z, relations = self.kgat_model.encode(x, edge_index, relations, return_attention_weights=False)

        reference_triples = self.kgat_model.teta_t(z, edge_index, relations)
        query = self.kgat_model.teta_q(query)

        source_batch = node_batch[edge_index[0]]
        tgt_batch = node_batch[edge_index[2]]
        
        if (source_batch != tgt_batch).any():
            raise ValueError(f"Intersection between batch, there are connection between different graph \n source_batch : {source_batch} \n tgt_batch : {tgt_batch}")
        
        reference_batch = source_batch

        gate = self.kgat_model.decode(reference_triples,
                             query,
                             value_batch=reference_batch,
                             query_batch=query_batch,
                             sigmoid=False)
        
        gate = softmax(gate, index=reference_batch, ptr=None, num_nodes=None, dim=0)
        virtual_token = self.reduce(gate * reference_triples, index=reference_batch, ptr=None, num_nodes=None, dim=0)

        virtual_token = self.graph_prefix(virtual_token)

        virtual_token = virtual_token.view(virtual_token.shape[0], n_token, self.language_model.embed_dim)

        return virtual_token
    
    def forward_lmkbc(self, x, edge_index, relations, query, node_batch, query_batch, n_token, input_ids, attention_mask, batch=None):
        virtual_token = self.forward(x, edge_index, relations, query, node_batch, query_batch, n_token)
        return self.language_model.forward_lmkbc(input_ids, attention_mask, virtual_token, batch)
    
    def generate_lmkbc(self, x, edge_index, relations, query, node_batch, query_batch, n_token, input_ids, attention_mask, batch=None, **kwargs):
        virtual_token = self.forward(x, edge_index, relations, query, node_batch, query_batch, n_token)
        return self.language_model.generate_lmkbc(input_ids, attention_mask, virtual_token, batch, **kwargs)
    
    def reduce(self, x, index = None,
               ptr = None, dim_size = None,
               dim = -2, reduce = 'sum'):

        if (ptr is not None and torch_geometric.typing.WITH_TORCH_SCATTER
                and not is_compiling()):
            ptr = expand_left(ptr, dim, dims=x.dim())
            return segment(x, ptr, reduce=reduce)

        if index is None:
            raise NotImplementedError(
                "Aggregation requires 'index' to be specified")
        return scatter(x, index, dim, dim_size, reduce)