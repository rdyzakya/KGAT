from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import json
from ._modeling_utils import read_txt

ALLOWED_SENTENCE_EMB = ["eol", "pcot", "ke"]

class LMKBCDataset(Dataset):
    def __init__(self,
                 data_path,
                 texts_txt_path,
                 entities_txt_path,
                 relations_txt_path,
                 entities_alias_path,
                 triples_path,
                 entities_tensor_path,
                 relations_tensor_path,
                 sentence_emb_mode="eol",
                 sentence_emb_index=None):
        self.ds = pd.read_json(data_path, lines=True)
        self.texts = read_txt(texts_txt_path)
        self.entities = read_txt(entities_txt_path)
        self.relations = read_txt(relations_txt_path)
        self.entities_alias = pd.read_json(entities_alias_path, lines=True)
        self.triples = np.array(json.load(open(triples_path, 'r')))

        sentence_emb_index = sentence_emb_index or -1
        
        entities_attr = torch.load(entities_tensor_path)[sentence_emb_mode]
        entities_attr = entities_attr[sentence_emb_index] if entities_attr.dim() == 3 else entities_attr
        assert entities_attr.dim() == 2
        self.entities_attr = entities_attr

        relations_attr = torch.load(relations_tensor_path)[sentence_emb_mode]
        relations_attr = relations_attr[sentence_emb_index] if relations_attr.dim() == 3 else relations_attr
        assert relations_attr.dim() == 2
        self.relations_attr = relations_attr

        self.sentence_emb_mode = sentence_emb_mode
        self.sentence_emb_index = sentence_emb_index
    
    def build_reference(self,
                        n_reference_min,
                        n_reference_max,
                        stay_ratio=1.0,
                        random_state=None):
        n_reference_low = n_reference_min
        n_reference_high = n_reference_max + 1

        state = np.random.get_state()
        if random_state:
            np.random.seed(random_state)

        all_triple_idx = np.array([i for i in range(len(self.triples))])

        reference = []
        for i, row in self.ds.iterrows():
            entry = {}
            triple = np.array(row["triple"])

            current_stay_ratio = np.random.rand() if stay_ratio == "random" else stay_ratio
            n_stay_triple = int(np.ceil(current_stay_ratio * len(triple)))

            stay_triple = np.random.choice(triple, n_stay_triple, replace=False)

            n_unrelated_triple = n_reference_min - n_stay_triple if n_reference_min == n_reference_max else np.random.randint(n_reference_low, n_reference_high) - n_stay_triple
            unrelated_triple = np.random.choice(
                all_triple_idx[~np.isin(all_triple_idx, triple)],
                n_unrelated_triple,
                replace=False
            )

            entry["triple_idx"] = np.concatenate((stay_triple, unrelated_triple)).tolist()

            link_cls_label = [1 for _ in range(len(stay_triple))] + [0 for _ in range(len(unrelated_triple))]

            coo_transpose = self.triples[entry["triple_idx"]]
            entities_index = np.concatenate((coo_transpose[:,0], coo_transpose[:,2]))
            entities_index = np.unique(entities_index)
        
        np.random.set_state(state)