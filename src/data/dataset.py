from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import json
import os
from ._data_utils import (
    read_txt
)

ALLOWED_SENTENCE_EMB = ["eol", "pcot", "ke"]

class KGATDataset(Dataset):
    def __init__(self,
                builder,
                texts_txt_path,
                entities_txt_path,
                relations_txt_path,
                entities_alias_path,
                texts_tensor_path=None,
                entities_tensor_path=None,
                relations_tensor_path=None,
                sentence_emb_mode="eol",
                sentence_emb_index=None):
        self.items = builder.items
        self.triples = builder.triples
        self.texts = read_txt(texts_txt_path)
        self.entities = read_txt(entities_txt_path)
        self.relations = read_txt(relations_txt_path)
        self.entities_alias = pd.read_json(entities_alias_path, lines=True)

        sentence_emb_index = sentence_emb_index or -1
        
        if texts_tensor_path is None:
            self.texts_attr = None
        else:
            texts_attr = torch.load(texts_tensor_path)[sentence_emb_mode]
            texts_attr = texts_attr[sentence_emb_index] if texts_attr.dim() == 3 else texts_attr
            assert texts_attr.dim() == 2
            self.texts_attr = texts_attr

        if entities_tensor_path is None:
            self.entities_attr = None
        else:
            entities_attr = torch.load(entities_tensor_path)[sentence_emb_mode]
            entities_attr = entities_attr[sentence_emb_index] if entities_attr.dim() == 3 else entities_attr
            assert entities_attr.dim() == 2
            self.entities_attr = entities_attr
        
        if relations_tensor_path is None:
            self.relations_attr = None
        else:
            relations_attr = torch.load(relations_tensor_path)[sentence_emb_mode]
            relations_attr = relations_attr[sentence_emb_index] if relations_attr.dim() == 3 else relations_attr
            assert relations_attr.dim() == 2
            self.relations_attr = relations_attr

        self.sentence_emb_mode = sentence_emb_mode
        self.sentence_emb_index = sentence_emb_index
    
    def lmkbc(self):
        pass

    def subgraph_gen(self):
        result = []
        for i, row in self.items.iterrows():
            entry = (

            )
            # injector(x, 
            # edge_index, 
            # relations, 
            # injection_node, 
            # # node_batch, 
            # # injection_node_batch)