import unittest
from model import (
    Injector,
    Detach,
    GATv2Encoder,
    InnerOuterProductDecoder,
    AutoModelForLMKBC
)
from transformers import AutoTokenizer
import torch
import random

class ModelTestCase(unittest.TestCase):

    def test_injector(self):
        N_NODE = random.randint(5,20)
        N_RELATION = random.randint(2,15)
        N_EDGE = random.randint(10,100)
        N_BATCH = random.randint(1,5)
        N_INJECTION_NODE = N_BATCH
        DIM = 768

        x = torch.randn(N_NODE, DIM)
        relations = torch.randn(N_RELATION, DIM)
        edge_index = torch.stack([
            torch.randint(0, N_NODE, (N_EDGE,)),
            torch.randint(0, N_RELATION, (N_EDGE,)),
            torch.randint(0, N_NODE, (N_EDGE,))
        ])

        injection_node = torch.randn(N_INJECTION_NODE, DIM)
        node_batch = torch.randint(0, N_BATCH, (N_NODE,))
        injection_node_batch = torch.arange(0,N_INJECTION_NODE)

        injector = Injector(edge_dim=DIM)
        x_out, edge_index_out, relations_out, x_is_injected, edge_is_injected, relations_is_injected = injector(x, edge_index, relations, injection_node, node_batch, injection_node_batch)

        self.assertEqual(x_out.shape[0], x.shape[0] + injection_node.shape[0])
        self.assertEqual(x_out.shape[1], x.shape[1])

        self.assertEqual(edge_index_out.shape[0], edge_index.shape[0])
        self.assertEqual(edge_index_out.shape[1], edge_index.shape[1] + x.shape[0])

        self.assertEqual(relations_out.shape[0], relations.shape[0] + 1)
        self.assertEqual(relations_out.shape[1], relations.shape[1])

        self.assertEqual(x_is_injected.shape[0], x_out.shape[0])
        self.assertEqual(edge_is_injected.shape[0], edge_index_out.shape[1])
        self.assertEqual(relations_is_injected.shape[0], relations_out.shape[0])

        detach = Detach()

        x_out, edge_index_out, relations_out = detach(x_out, edge_index_out, relations_out, x_is_injected, edge_is_injected, relations_is_injected)

        self.assertTrue((x_out == x).all())
        self.assertTrue((edge_index_out == edge_index).all())
        self.assertTrue((relations_out == relations).all())
    
    def test_gae(self):
        N_NODE = random.randint(5,20)
        N_RELATION = random.randint(2,15)
        N_EDGE = random.randint(10,100)
        N_LAYERS = random.randint(1,5)
        N_HEAD = random.randint(1,8)
        DIM = 768
        H_DIM = 128

        x = torch.randn(N_NODE, DIM)
        relations = torch.randn(N_RELATION, DIM)
        edge_index = torch.stack([
            torch.randint(0, N_NODE, (N_EDGE,)),
            torch.randint(0, N_RELATION, (N_EDGE,)),
            torch.randint(0, N_NODE, (N_EDGE,))
        ])

        encoder = GATv2Encoder(in_channels=DIM, hidden_channels=H_DIM, num_layers=N_LAYERS, heads=N_HEAD)
        decoder = InnerOuterProductDecoder(num_features=DIM)

        z = encoder(x, edge_index, relations)
        adj = decoder.forward_all(z, relations, sigmoid=False)
        adj_not_all = decoder(z, edge_index, relations, sigmoid=False)

        self.assertEqual(z.shape[0], x.shape[0])
        self.assertEqual(z.shape[1], x.shape[1])

        self.assertEqual(adj.shape[0], relations.shape[0])
        self.assertEqual(adj.shape[1], x.shape[0])
        self.assertEqual(adj.shape[2], x.shape[0])

        self.assertEqual(adj_not_all.shape[0], edge_index.shape[1])
    
    def test_lm(self):
        model_name_or_path = "openai-community/gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
        model = AutoModelForLMKBC.from_pretrained(model_name_or_path, device_map="auto")

        model.prepare_tokenizer(tokenizer)

        text = [
            "Hello world",
            "Lorem ipsum dolor sit amet",
            "My number one text",
            "Test"
        ]

        tokenized = tokenizer(text, padding=True, return_tensors="pt")
        tokenized.to(model.device)

        text_emb = model.text_embedding(index=-1, **tokenized)

        self.assertEqual(text_emb.shape[0], len(text))


        

if __name__ == '__main__':
    unittest.main()