from .lm import AutoModelForLMKBC
from .injector import Injector, Detach
from .gae import KGATEncoder, KGATRetrieval, KGATTripleEmb, KGATModel
from .aggr import AttentionalAggregation, SoftmaxAggregation
from .base_model import BaseModel
from .graph_prefix import GraphPrefix

from torch_geometric.utils import softmax

class Pipeline(BaseModel):
    def __init__(self, kgat_model, graph_prefix, language_model):
        super().__init__()
        self.kgat_model = kgat_model
        self.graph_prefix = graph_prefix
        self.language_model = language_model
        # self.readout = AttentionalAggregation()
    
    def forward(self, x, edge_index, relations, query, node_batch, query_batch, input_ids, attention_mask):
        # z, relations = self.kgat_model.encode(x, edge_index, relations, return_attention_weights=False)

        # reference_triples = self.teta_t(z, edge_index, relations)


        # query = self.kgat_model.teta_q(query)

        kgat_out = self.kgat_model(x,
                edge_index, 
                relations, 
                query,
                values=None,
                node_batch=node_batch, 
                query_batch=query_batch,
                values_batch=None,
                sigmoid=False,
                allow_intersection=False)
        
        source_batch = node_batch[edge_index[0]]
        tgt_batch = node_batch[edge_index[2]]
        
        if (source_batch != tgt_batch).any():
            raise ValueError(f"Intersection between batch, there are connection between different graph \n source_batch : {source_batch} \n tgt_batch : {tgt_batch}")
        
        reference_batch = source_batch

        # gate = softmax(gate, index, ptr, dim_size, dim)
        # out = self.reduce(gate * x, index, ptr, dim_size, dim)

        gate = softmax(kgat_out, index=reference_batch, ptr=None, num_nodes=None, dim=0)
        