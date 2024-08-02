import unittest
from model import (
    Injector,
    Detach,
    GATv2Encoder,
    InnerOuterProductDecoder,
    NodeClassifierDecoder,
    MultiheadGAE,
    AttentionalAggregation,
    SoftmaxAggregation,
    GraphPrefix,
    AutoModelForLMKBC
)
from transformers import AutoTokenizer
import torch
from torch_geometric.nn.dense.linear import Linear
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

        (x_out, edge_index_out, relations_out,
         injection_node, _, injection_relation) = detach(x_out, edge_index_out, relations_out, x_is_injected, edge_is_injected, relations_is_injected)

        self.assertTrue((x_out == x).all())
        self.assertTrue((edge_index_out == edge_index).all())
        self.assertTrue((relations_out == relations).all())
    
    def test_aggr(self):
        N_BATCH = random.randint(1,5)
        N_NODE = random.randint(5,20)
        DIM = 768
        OUT_DIM = DIM

        node_batch = torch.randint(0, N_BATCH, (N_NODE,))

        x = torch.randn(N_NODE, DIM)

        gate_nn = Linear(in_channels=DIM, out_channels=OUT_DIM, bias=False, weight_initializer="glorot") # untuk ngeliat importance, nanti direduce aja dim = 1 (sum)
        nn = torch.nn.Identity()

        attentional_aggr = AttentionalAggregation(gate_nn=gate_nn, nn=nn)
        softmax_aggr = SoftmaxAggregation(learn=True, channels=1)

        out_attention, gate = attentional_aggr(x, index=node_batch, return_gate=True)
        out_softmax, alpha = softmax_aggr(x, index=node_batch, return_alpha=True)

        self.assertEqual(out_attention.shape[0], N_BATCH)
        self.assertEqual(out_softmax.shape[0], N_BATCH)

        self.assertEqual(out_attention.shape[1], OUT_DIM)
        self.assertEqual(out_softmax.shape[1], x.shape[1])

        self.assertEqual(gate.shape[0], N_NODE)
        self.assertEqual(alpha.shape[0], N_NODE)

        self.assertEqual(gate.shape[1], OUT_DIM)
        self.assertEqual(alpha.shape[1], DIM)

    
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
    
    def test_mhgae(self):
        N_NODE = random.randint(5,20)
        N_RELATION = random.randint(2,15)
        N_EDGE = random.randint(10,100)
        N_LAYERS = random.randint(1,5)
        N_HEAD = random.randint(1,8)
        DIM = 768
        H_DIM = 128
        N_BATCH = random.randint(1,5)
        N_INJECTION_NODE = N_BATCH

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

        mhgae = MultiheadGAE(in_channels=DIM, hidden_channels=H_DIM, num_layers=N_LAYERS, heads=N_HEAD, subgraph=True)

        z, all_adj, all_alpha, out_link, out_node = mhgae.forward(x, 
                                                        edge_index, 
                                                        relations, 
                                                        injection_node=injection_node, 
                                                        node_batch=node_batch, 
                                                        injection_node_batch=injection_node_batch, 
                                                        return_attention_weights=True, 
                                                        all=False, 
                                                        sigmoid=False)
        
        self.assertEqual(z.shape[0], x.shape[0])
        self.assertEqual(z.shape[1], x.shape[1])

        self.assertEqual(out_link.shape[0], edge_index.shape[1])

        self.assertEqual(out_node.shape[0], z.shape[0])
        self.assertEqual(out_node.shape[1], 1)
        

    def test_gprefix(self):
        N_NODE = random.randint(5,20)
        N_LAYERS = random.randint(1,5)
        N_TOKENS = random.randint(1,5)
        N_BATCH = random.randint(1,5)
        N_INJECTION_NODE = N_BATCH
        DIM = 768
        H_DIM = 512

        injection_node = torch.randn(N_INJECTION_NODE, DIM)
        node_batch = torch.randint(0, N_BATCH, (N_NODE,))
        injection_node_batch = torch.arange(0,N_INJECTION_NODE)

        x = torch.randn(N_NODE, DIM)

        node_decoder = NodeClassifierDecoder()

        gprefix = GraphPrefix(num_features=DIM, hidden_channels=H_DIM, num_layers=N_LAYERS, n_tokens=N_TOKENS)

        attentional_aggr = AttentionalAggregation(gate_nn=node_decoder, nn=gprefix.nn)

        out_attention, gate = attentional_aggr(x, 
                                               index=node_batch, 
                                               return_gate=True, 
                                               injection_node=injection_node, 
                                               node_batch=node_batch, 
                                               injection_node_batch=injection_node_batch, 
                                               sigmoid=False)

        self.assertEqual(out_attention.shape[0], N_BATCH)

        self.assertEqual(out_attention.shape[1], N_TOKENS*DIM)

        self.assertEqual(gate.shape[0], N_NODE)

        self.assertEqual(gate.shape[1], 1)

if __name__ == '__main__':
    unittest.main()