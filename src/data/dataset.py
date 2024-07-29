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

class KGAT(Dataset):
    def __init__(self,
                texts_txt_path,
                entities_txt_path,
                relations_txt_path,
                entities_alias_path,
                # triples_path,
                entities_tensor_path,
                relations_tensor_path,
                sentence_emb_mode="eol",
                sentence_emb_index=None,
                n_reference_min=30,
                n_reference_max=50,
                stay_ratio_min=1.0,
                stay_ratio_max=1.0,
                random_state=None,
                n_pick=1,
                items_path="./items.jsonl",
                save_items=False):
        # self.ds = pd.read_json(data_path, lines=True)
        self.texts = read_txt(texts_txt_path)
        self.entities = read_txt(entities_txt_path)
        self.relations = read_txt(relations_txt_path)
        self.entities_alias = pd.read_json(entities_alias_path, lines=True)
        # self.triples = np.array(json.load(open(triples_path, 'r')))

        sentence_emb_index = sentence_emb_index or -1
        
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

        if not os.path.exists(items_path):
            items = self.build(n_reference_min=n_reference_max,
                                    n_reference_max=n_reference_min,
                                    stay_ratio_min=stay_ratio_min,
                                    stay_ratio_max=stay_ratio_max,
                                    random_state=random_state,
                                    n_pick=n_pick)
            items = pd.DataFrame(items)
            self.items = items
        else:
            self.items = pd.read_json(items_path, orient="records", lines=True)
        if save_items:
            self.items.to_json(items_path, orient="records", lines=True)