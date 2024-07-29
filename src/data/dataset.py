from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import json
from ._data_utils import (
    read_txt,
    bounded_random
)

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
    
    def _choice(self, a, size):
        if size > len(a):
            return a
        else:
            return np.random.choice(a, size, replace=False)

    def _pick_idx(self,
                  all_triple_idx,
                  triple_idx,
                  stay_triple_idx,
                  reference_idx,
                  n_pick=1):
        unrelated_triple_idx = all_triple_idx[~np.isin(all_triple_idx, triple_idx) & ~np.isin(all_triple_idx, reference_idx)]
        prob = np.random.rand()
        if prob <= 0.35: # 35% related node
            reference_triple = self.triples[reference_idx]
            reference_subject = reference_triple[:,0]
            reference_object = reference_triple[:,2]
            reference_entities = np.unique(np.concatenate((reference_subject, reference_object)))

            unrelated_triple = self.triples[unrelated_triple_idx]
            unrelated_subject = unrelated_triple[:,0]
            unrelated_object = unrelated_triple[:,2]

            isin_reference = np.isin(unrelated_subject, reference_entities) | np.isin(unrelated_object, reference_entities)

            picked_idx = self._choice(unrelated_triple_idx[isin_reference], n_pick)

            return picked_idx
        elif prob <= 0.85: # 50% related relation
            # 60% same relation with stay triple
            # 40% with reference
            relation_prob = np.random.rand()
            if relation_prob <= 0.6:
                stay_triple = self.triples[stay_triple_idx]
                stay_relations = np.unique(stay_triple[:,1])

                unrelated_triple = self.triples[unrelated_triple_idx]
                unrelated_relations = unrelated_triple[:,1]

                isin_reference = np.isin(unrelated_relations, stay_relations)

                picked_idx = self._choice(unrelated_triple_idx[isin_reference], n_pick)

                return picked_idx
            else:
                reference_triple = self.triples[reference_idx]
                reference_relations = np.unique(reference_triple[:,1])

                unrelated_triple = self.triples[unrelated_triple_idx]
                unrelated_relations = unrelated_triple[:,1]

                isin_reference = np.isin(unrelated_relations, reference_relations)

                picked_idx = self._choice(unrelated_triple_idx[isin_reference], n_pick)
                return picked_idx
        else: # 15% random
            picked_idx = self._choice(unrelated_triple_idx, n_pick)
            return picked_idx
    
    def build_reference(self,
                        n_reference_min,
                        n_reference_max,
                        stay_ratio_min=1.0,
                        stay_ratio_max=1.0,
                        random_state=None,
                        n_pick=1):
        # 40% take from same relation, 40% take from related nodes, 20% random nodes

        state = np.random.get_state()
        if random_state:
            np.random.seed(random_state)

        all_triple_idx = np.array([i for i in range(len(self.triples))])

        reference = []
        for i, row in self.ds.iterrows():
            entry = {}
            triple_idx = np.array(row["triple"])

            # remove arbitrary
            n_stay_min = int(np.ceil(stay_ratio_min * len(triple_idx)))
            n_stay_max = int(np.ceil(stay_ratio_max * len(triple_idx)))
            n_stay_triple = bounded_random(n_stay_min, n_stay_max)
            stay_triple_idx = np.random.choice(triple_idx, n_stay_triple, replace=False)

            reference_idx = stay_triple_idx.copy()

            n_reference = bounded_random(n_reference_min, n_reference_max)

            n_unrelated_triple = n_reference - n_stay_triple

            unrelated_triple_idx = all_triple_idx[~np.isin(all_triple_idx, triple_idx)]
            # picking index with certain conditions
            if len(unrelated_triple_idx) <= n_unrelated_triple:
                reference_idx = np.concatenate((reference_idx, unrelated_triple_idx))
            else:
                while len(reference_idx) < n_reference:
                    picked_idx = self._pick_idx(all_triple_idx, triple_idx, stay_triple_idx, reference_idx, n_pick=n_pick)
                    reference_idx = np.concatenate((reference_idx, picked_idx))
            
            # create link classification label
            link_cls_label = [1 if i < n_stay_triple else 0 in range(len(reference_idx))]
            # create node classification label
            
        
        np.random.set_state(state)