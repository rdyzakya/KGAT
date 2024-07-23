import unittest
from kgat.model import (
    GraphEncoderDecoder,
    SubgraphGenerator,
    VirtualTokenGenerator,
    load_model_lmkbc
)

from kgat.model.graph import (
    RESCAL,
    GraphEncoder,
    Injector,
    ReshapeRelation,
    VirtualToken
)

import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class ModelModule(unittest.TestCase):

    def test_decoder(self):
        entities = torch.randn(256, 768) # 256 entities with 768 dimension
        relations = torch.randn(16, 768) # 16 relations with 768 * 768 dimension

        decoder = RESCAL(n_features=768)

        scores = decoder(entities, relations)

        self.assertEqual(scores.shape[0],entities.shape[0])
        self.assertEqual(scores.shape[1],relations.shape[0])
        self.assertEqual(scores.shape[2],entities.shape[0])

        print(f"TEST DECODER DIM : 768 --> {count_parameters(decoder)}")

    def test_encoder(self):
        n_features = 768
        h_dim = 512
        n_head = 3
        p = 0.2
        n_layers = 4

        encoder = GraphEncoder(n_features=n_features, h_dim=h_dim, n_head=n_head, p=p, n_layers=n_layers)

        entities = torch.randn(256, n_features)
        relations = torch.randn(16, n_features)

        edge_index = torch.randint(256, (2, 64))
        relation_index = torch.randint(16, (64,))

        out = encoder(entities, edge_index, relations, relation_index)

        self.assertEqual(out[0].shape[0], entities.shape[0])
        self.assertEqual(out[0].shape[1], n_features)
        self.assertEqual(out[1].shape[0], relations.shape[0])
        self.assertEqual(out[1].shape[1], n_features)

        print(f"TEST ENCODER DIM : 768 | HEAD : 3 | N_LAYERS : 4 --> {count_parameters(encoder)}")
    
    def test_injector(self):
        input_dim = 768
        h_dim = 1024
        n_head=2

        queries = torch.randn(8, input_dim) # 8 queries with 768 dimension
        entities = torch.randn(256, input_dim) # 256 entities with 768 dimension
        relations = torch.randn(16, input_dim) # 16 relation types with 768 dimension

        injector = Injector(input_dim=input_dim,
                                    n_head=n_head,
                                    p=0.0)
        
        edge_index = torch.randint(256, (2,64))
        relation_index = torch.randint(16, (64,))
        batch = torch.arange(0, 8).repeat(256//8)
        batch, _ = batch.sort()
        
        out_node, out_edge = injector(queries, 
                                    entities, 
                                    edge_index, 
                                    relations, 
                                    relation_index, 
                                    batch)

        self.assertEqual(out_node.shape, entities.shape)
        self.assertEqual(out_edge.shape, relations.shape)

        print(f"TEST INJECTOR INPUT_DIM : 768 | N_HEAD : 2 --> {count_parameters(injector)}")
    
    
    def test_virtual_token(self):
        n_virtual_token = 3
        n_features = 768

        virtual_token = VirtualToken(n_features, n_virtual_token)

        entities = torch.randn(16, n_features) # 16 entities
        relations = torch.randn(4, n_features) # 4 relations

        batch_index = torch.arange(16)
        batch_index = torch.tensor([
            0,0,0,0,
            1,1,1,1,
            2,2,2,2,
            3,3,3,3,
        ])

        out = virtual_token(entities, batch_index=batch_index)

        self.assertEqual(out.shape[0], batch_index.max()+1)
        self.assertEqual(out.shape[1], n_virtual_token)
        self.assertEqual(out.shape[2], entities.shape[1])

        print(f"TEST VIRTUAL TOKEN DIM : 768 | N_VT : 3 --> {count_parameters(virtual_token)}")
    
    def test_all(self):
        # input_dim = 768
        # encoder_decoder_h_dim = 1024
        # out_dim = 768
        # reshape_h_dim = 128
        n_features = 768
        h_dim = 512
        n_injector_head = 2
        injector_dropout_p = 0.2
        encoder_dropout_p = 0.2
        n_encoder_head = 3
        n_encoder_layers = 2
        n_virtual_token = 3

        queries = torch.randn(8, n_features) # 8 queries
        entities = torch.randn(256, n_features) # 256 entities
        relations = torch.randn(16, n_features) # 16 relations (relation type)

        edge_index = torch.randint(256, (2, 64))
        relation_index = torch.randint(16, (64,))

        x_coo = torch.vstack([edge_index[0], relation_index, edge_index[1]])
        x_coo = x_coo.transpose(0,1)

        batch = torch.arange(0, 8).repeat(256//8)

        subgenerator = SubgraphGenerator(n_features=n_features,
                                         h_dim=h_dim,
                                        n_injector_head=n_injector_head,
                                        injector_dropout_p=injector_dropout_p, 
                                        encoder_dropout_p=encoder_dropout_p,
                                        n_encoder_head=n_encoder_head,
                                        n_encoder_layers=n_encoder_layers)
        
        out_encoder_decoder = subgenerator.encoder_decoder(entities, relations, x_coo)

        self.assertEqual(out_encoder_decoder.shape[0], entities.shape[0])
        self.assertEqual(out_encoder_decoder.shape[1], relations.shape[0])
        self.assertEqual(out_encoder_decoder.shape[2], entities.shape[0])

        out_subgenerator = subgenerator(queries, entities, relations, x_coo, batch)

        self.assertEqual(out_subgenerator.shape[0], entities.shape[0])
        self.assertEqual(out_subgenerator.shape[1], relations.shape[0])
        self.assertEqual(out_subgenerator.shape[2], entities.shape[0])

        vt_generator = VirtualTokenGenerator.from_subgraph_generator(subgenerator, n_virtual_token=n_virtual_token)

        print("VT Generator params :", sum(p.numel() for p in vt_generator.parameters()))

        vt_out = vt_generator(queries, entities, relations, x_coo, batch)

        self.assertEqual(vt_out.shape[0], queries.shape[0])
        self.assertEqual(vt_out.shape[1], n_virtual_token)
        self.assertEqual(vt_out.shape[2], queries.shape[1])

        print(f"TEST SG --> {count_parameters(subgenerator)}")
        print(f"TEST VTGEN --> {count_parameters(vt_generator)}")

if __name__ == '__main__':
    unittest.main()