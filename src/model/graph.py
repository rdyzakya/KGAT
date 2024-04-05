import torch
from llama import Transformer, Tokenizer

class SubgraphPooler(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = None
        self.lm = None
        self.subgraph_pooler = None
    
    def forward(self, text_input_ids, entities_input_ids, relations_input_ids, x_coo, coo_mask=None):
        pass

class SubgraphGenerator(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.subgraph_pooler = SubgraphPooler()
        self.decoder = None
    
    def forward(self, text_input_ids, entities_input_ids, relations_input_ids, x_coo, coo_mask=None):
        pass

class SubgraphVTTransformer(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        pass