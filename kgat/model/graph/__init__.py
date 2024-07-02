from .inject import (
    Injector
)
from .decoder import (
    RESCAL
)
from .encoder import (
    GraphEncoder
)
from .relation import (
    ReshapeRelation
)
from .virtual_token import (
    VirtualToken
)
import torch
import warnings

class GraphEncoderDecoder(torch.nn.Module):
    # def __init__(self, input_dim=768, 
    #              h_dim=1024, 
    #              out_dim=768, 
    #              reshape_h_dim=128,
    #              encoder_dropout_p=0.2, 
    #              n_encoder_head=1, 
    #              n_encoder_layers=1):
    def __init__(self, dim=768, 
                 encoder_dropout_p=0.2, 
                 n_encoder_head=1, 
                 n_encoder_layers=1):
        super().__init__()
        self.encoder = GraphEncoder(dim=dim,
                                    n_head=n_encoder_head,
                                    p=encoder_dropout_p,
                                    n_layers=n_encoder_layers)
        self.relation = ReshapeRelation(input_dim=dim)
        self.decoder = RESCAL()
    
    def forward(self, entities, relations, x_coo):
        # x_coo shape is N_triplets * 3
        edge_index = x_coo[:, [0,2]].transpose(0, 1)
        relation_index = x_coo[:, 1]

        entities_emb, relations_emb = self.encoder(entities, edge_index, relations, relation_index)
        relation_matrices = self.relation(relations_emb)
        score = self.decoder(entities_emb, relation_matrices)
        return score

class SubgraphGenerator(torch.nn.Module):
    # def __init__(self, input_dim=768, 
    #              encoder_decoder_h_dim=1024, 
    #              out_dim=768, 
    #              reshape_h_dim=128,
    #              n_injector_head=1, 
    #              injector_dropout_p=0.2, 
    #              encoder_dropout_p=0.2, 
    #              n_encoder_head=1, 
    #              n_encoder_layers=1):
    def __init__(self, dim=768,
                 n_injector_head=1, 
                 injector_dropout_p=0.2, 
                 encoder_dropout_p=0.2, 
                 n_encoder_head=1, 
                 n_encoder_layers=1):
        super().__init__()
        self.injector = Injector(input_dim=dim,
                                n_head=n_injector_head,
                                p=injector_dropout_p)
        self.encoder_decoder = GraphEncoderDecoder(dim=dim,
                                                   encoder_dropout_p=encoder_dropout_p,
                                                   n_encoder_head=n_encoder_head, 
                                                   n_encoder_layers=n_encoder_layers)

        # self.input_dim = input_dim 
        # self.encoder_decoder_h_dim = encoder_decoder_h_dim 
        # self.out_dim = out_dim 
        # self.reshape_h_dim = reshape_h_dim
        self.dim = dim
        self.n_injector_head = n_injector_head
        self.injector_dropout_p = injector_dropout_p
        self.encoder_dropout_p = encoder_dropout_p
        self.n_encoder_head = n_encoder_head
        self.n_encoder_layers = n_encoder_layers
    
    def forward(self, queries, entities, relations, x_coo, batch):
        edge_index = x_coo[:, [0,2]].transpose(0, 1)
        relation_index = x_coo[:, 1]

        injected_entities, injected_relations = self.injector(queries, entities, edge_index, relations, relation_index, batch)
        score = self.encoder_decoder(injected_entities, injected_relations, x_coo)
        return score

class VirtualTokenGenerator(torch.nn.Module):
    def from_subgraph_generator(subgraph_generator : SubgraphGenerator, n_virtual_token):
        injector = subgraph_generator.injector
        encoder = subgraph_generator.encoder_decoder.encoder
        dim = subgraph_generator.dim
        return VirtualTokenGenerator(injector=injector, encoder=encoder,
                                     n_virtual_token=n_virtual_token, dim=dim)

    # def __init__(self, input_dim=768,
    #              encoder_h_dim=1024, 
    #              out_dim=768, 
    #              n_injector_head=1, 
    #              n_encoder_head=1, 
    #              injector_dropout_p=0.2,
    #              encoder_dropout_p=0.2, 
    #              n_encoder_layers=1,
    #              n_virtual_token=3,
    #              injector=None,
    #              encoder=None):
    def __init__(self, dim=768,
                 n_injector_head=1, 
                 n_encoder_head=1, 
                 injector_dropout_p=0.2,
                 encoder_dropout_p=0.2, 
                 n_encoder_layers=1,
                 n_virtual_token=3,
                 injector=None,
                 encoder=None):
        super().__init__()
        self.injector = injector or Injector(input_dim=dim,
                                            n_head=n_injector_head,
                                            p=injector_dropout_p)
        self.encoder = encoder or GraphEncoder(dim=dim,
                                    n_head=n_encoder_head,
                                    p=encoder_dropout_p,
                                    n_layers=n_encoder_layers)
        self.virtual_token = VirtualToken(n_virtual_token=n_virtual_token,
                                          n_features=dim)
        self.n_object_predictor = torch.nn.Linear(dim * n_virtual_token, 1, bias=True)

        # self.input_dim = self.injector.input_dim
        # self.encoder_h_dim = self.encoder.h_dim
        # self.out_dim = self.encoder.h_dim
        self.dim = self.injector.input_dim
        self.n_injector_head = self.injector.n_head
        self.n_encoder_head = self.encoder.n_head
        self.injector_dropout_p = self.injector.p
        self.encoder_dropout_p = self.encoder.p
        self.n_encoder_layers = self.encoder.n_layers
        self.n_virtual_token = self.virtual_token.n_virtual_token

        
    def forward(self, queries, entities, relations, x_coo, batch):
        edge_index = x_coo[:, [0,2]].transpose(0, 1)
        relation_index = x_coo[:, 1]

        src_batch = batch[edge_index[0]]
        tgt_batch = batch[edge_index[1]]
        intersection_condition = src_batch != tgt_batch

        if intersection_condition.any():
            warnings.warn(f"There are intersections in the graph, please check node {intersection_condition.nonzero()}")

        injected_entities, injected_relations = self.injector(queries, entities, edge_index, relations, relation_index, batch)

        entities_emb, relations_emb = self.encoder(injected_entities, edge_index, injected_relations, relation_index)

        out_vt = self.virtual_token(entities_emb, batch)

        out_n_object = self.n_object_predictor(out_vt.reshape(-1, out_vt.shape[1] * out_vt.shape[2]))
        out_n_object = out_n_object.relu() # >= 0

        return out_vt, out_n_object
    
    def freeze_injector_and_encoder(self):
        for param in self.injector.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

if __name__ == "__main__":
    input_dim = 768
    encoder_decoder_h_dim = 1024
    out_dim = 768
    reshape_h_dim = 128
    n_injector_head = 2
    injector_dropout_p = 0.2
    encoder_dropout_p = 0.2
    n_encoder_head = 3
    n_encoder_layers = 2
    n_virtual_token = 3

    queries = torch.randn(8, input_dim) # 8 queries
    entities = torch.randn(256, input_dim) # 256 entities
    relations = torch.randn(16, input_dim) # 16 relations (relation type)

    edge_index = torch.randint(256, (2, 64))
    relation_index = torch.randint(16, (64,))

    x_coo = torch.vstack([edge_index[0], relation_index, edge_index[1]])
    x_coo = x_coo.transpose(0,1)

    batch = torch.arange(0, 8).repeat(256//8)

    subgenerator = SubgraphGenerator(input_dim=input_dim,
                                     encoder_decoder_h_dim=encoder_decoder_h_dim,
                                     out_dim=out_dim,
                                     reshape_h_dim=reshape_h_dim,
                                     n_injector_head=n_injector_head,
                                     injector_dropout_p=injector_dropout_p, 
                                     encoder_dropout_p=encoder_dropout_p,
                                     n_encoder_head=n_encoder_head,
                                     n_encoder_layers=n_encoder_layers)
    
    print("Subgenerator params :", sum(p.numel() for p in subgenerator.parameters()))
    
    out_encoder_decoder = subgenerator.encoder_decoder(entities, relations, x_coo)

    assert out_encoder_decoder.shape[0] == entities.shape[0]
    assert out_encoder_decoder.shape[1] == relations.shape[0]
    assert out_encoder_decoder.shape[2] == entities.shape[0]

    out_subgenerator = subgenerator(queries, entities, relations, x_coo, batch)

    assert out_subgenerator.shape[0] == entities.shape[0]
    assert out_subgenerator.shape[1] == relations.shape[0]
    assert out_subgenerator.shape[2] == entities.shape[0]

    vt_generator = VirtualTokenGenerator.from_subgraph_generator(subgenerator, n_virtual_token=n_virtual_token)

    print("VT Generator params :", sum(p.numel() for p in vt_generator.parameters()))

    out_vt_generator = vt_generator(queries, entities, relations, x_coo, batch)

    assert out_vt_generator.shape[0] == queries.shape[0]
    assert out_vt_generator.shape[1] == n_virtual_token
    assert out_vt_generator.shape[2] == queries.shape[1]