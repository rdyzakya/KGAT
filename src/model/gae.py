from torch_geometric.nn import TransformerConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.aggr import MeanAggregation
import torch
from torch.nn import ReLU, Sequential, LayerNorm, ModuleList
from torch_geometric.nn.dense import Linear
from .base_model import BaseModel

class KGATEncoderBlock(BaseModel):
    def __init__(self,
                 d_model,
                 d_ff,
                 heads=1,
                 beta=False,
                 dropout=0.0,
                 bias=True,
                 transform_relation=False,
                 **kwargs):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            beta=beta,
            dropout=dropout,
            bias=bias,
            transform_relation=transform_relation,
            **kwargs
        )
        d_v = d_model // heads
        self.att_gnn = TransformerConv(
            in_channels=d_model,
            out_channels=d_v,
            heads=heads,
            concat=True,
            beta=beta,
            dropout=dropout,
            edge_dim=d_model,
            bias=bias,
            root_weight=True,
            **kwargs
        )
        self.lin_o = Linear(in_channels=d_v*heads, out_channels=d_model)
        self.layer_norm1 = LayerNorm(normalized_shape=d_model)
        self.ffn = Sequential(
            Linear(in_channels=d_model, out_channels=d_ff),
            ReLU(),
            Linear(in_channels=d_ff, out_channels=d_model)
        )
        self.layer_norm2 = LayerNorm(normalized_shape=d_model)
    
    def forward(self, x, edge_index, relations, return_attention_weights=False):
        edge_attr = relations[edge_index[1]]
        ei = edge_index[[0,2]].long()

        out = self.att_gnn(x, ei, edge_attr, return_attention_weights=return_attention_weights if return_attention_weights else None)
        if return_attention_weights:
            out, (ei, alpha) = out
        
        out = self.lin_o(out) + x # skip conn
        out = self.layer_norm1(out)

        out = self.ffn(out) + out # skip conn
        out = self.layer_norm2(out)

        if self.transform_relation:
            out_rel = self.att_gnn.lin_skip(relations)
            out_rel = self.lin_o(out_rel) + relations
            out_rel = self.layer_norm1(out_rel)

            out_rel = self.ffn(out_rel) + out_rel
            out_rel = self.layer_norm2(out_rel)

            relations = out_rel

        if return_attention_weights:
            return out, relations, (ei, alpha)
        else:
            return out, relations

class KGATEncoder(BaseModel):
    def __init__(self,
                 d_model,
                 d_ff,
                 heads=1,
                 beta=False,
                 dropout=0.0,
                 bias=True,
                 transform_relation=False,
                 num_block=1,
                 **kwargs):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            beta=beta,
            dropout=dropout,
            bias=bias,
            transform_relation=transform_relation,
            num_block=num_block,
            **kwargs
        )

        self.encoder = ModuleList([
            KGATEncoderBlock(d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            beta=beta,
            dropout=dropout,
            bias=bias,
            transform_relation=transform_relation,
            **kwargs) for _ in range(num_block)
        ])
    
    def forward(self, x, edge_index, relations, return_attention_weights=False):
        z = x

        for enc in self.encoder:
            out = enc(z, edge_index, relations, return_attention_weights=return_attention_weights)
            if return_attention_weights:
                z, relations, (ei, alpha) = out
            else:
                z, relations = out
        if return_attention_weights:
            return z, relations, (ei, alpha) 
        else:
            return z, relations

class KGATTripleEmb(BaseModel):
    def __init__(self, d_model, bias=True):
        super().__init__(d_model=d_model, bias=bias)
        self.lin = Linear(in_channels=d_model*3, out_channels=d_model, bias=bias, weight_initializer="glorot")
    
    def forward(self, z, edge_index, relations):
        inputs = torch.hstack([
            z[edge_index[0]],
            relations[edge_index[1]],
            z[edge_index[2]]
        ])
        return self.lin(inputs)

class KGATRetrieval(BaseModel):
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

class KGATModel(BaseModel):
    def __init__(self,
                 d_model,
                 d_ff,
                 heads=1,
                 beta=False,
                 dropout=0.0,
                 bias=True,
                 transform_relation=False,
                 num_block=1,
                 **kwargs):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            beta=beta,
            dropout=dropout,
            bias=bias,
            transform_relation=transform_relation,
            num_block=num_block,
            **kwargs
        )

        self.encoder = KGATEncoder(
            d_model=d_model,
            d_ff=d_ff,
            heads=heads,
            beta=beta,
            dropout=dropout,
            bias=bias,
            transform_relation=transform_relation,
            num_block=num_block,
            **kwargs
        )

        self.teta_q = Linear(in_channels=d_model,
                            out_channels=d_model,
                            bias=bias,
                            weight_initializer="glorot")
        
        self.teta_t = KGATTripleEmb(d_model=d_model,
                                    bias=bias)

        self.decoder = KGATRetrieval()
    
    def encode(self,
               x,
               edge_index,
               relations,
               return_attention_weights=False):
        out = self.encoder(x, edge_index, relations, return_attention_weights=return_attention_weights)
        return out
    
    def decode(self,
               value, 
               query, 
               value_batch=None, 
               query_batch=None, 
               sigmoid=False):
        return self.decoder(value, query, value_batch=value_batch, query_batch=query_batch, sigmoid=sigmoid)
    
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
        
        z, relations = self.encode(x, edge_index, relations, return_attention_weights=False)

        reference_triples = self.teta_t(z, edge_index, relations)

        source_batch = node_batch[edge_index[0]]
        tgt_batch = node_batch[edge_index[2]]
        
        if (source_batch != tgt_batch).any() and not allow_intersection:
            raise ValueError(f"Intersection between batch, there are connection between different graph \n source_batch : {source_batch} \n tgt_batch : {tgt_batch}")
        
        reference_batch = source_batch

        qr_out = self.decode(reference_triples,
                             query,
                             value_batch=reference_batch,
                             query_batch=query_batch,
                             sigmoid=sigmoid)
        
        out = (qr_out,)

        query = self.teta_q(query)
        
        if values is not None:
            if len(values) > 0:
                rv_out = self.decode(reference_triples,
                             values,
                             value_batch=reference_batch,
                             query_batch=values_batch,
                             sigmoid=sigmoid)
                
                qv_out = self.decode(query,
                             values,
                             value_batch=query_batch,
                             query_batch=values_batch,
                             sigmoid=sigmoid)

                out += (rv_out, qv_out)
        
        return out[0] if len(out) == 1 else out