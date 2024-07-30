# from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import json
import os
from ._data_utils import (
    bounded_random
)

ALLOWED_SENTENCE_EMB = ["baseline", "eol", "pcot", "ke"]

class DSBuilder:
    def __init__(self,
                triples_path,
                data_path=None,
                n_reference_min=30,
                n_reference_max=50,
                stay_ratio_min=1.0,
                stay_ratio_max=1.0,
                random_state=None,
                n_pick=1,
                items_path="./items.jsonl",
                save_items=False):
        self.ds = pd.read_json(data_path, lines=True) if data_path is not None else None
        self.triples = np.array(json.load(open(triples_path, 'r')))
        if not os.path.exists(items_path) and data_path is not None:
            items = self.build(n_reference_min=n_reference_max,
                                    n_reference_max=n_reference_min,
                                    stay_ratio_min=stay_ratio_min,
                                    stay_ratio_max=stay_ratio_max,
                                    random_state=random_state,
                                    n_pick=n_pick)
            items = pd.DataFrame(items)
            self.items = items
        elif os.path.exists(items_path):
            self.items = pd.read_json(items_path, orient="records", lines=True)
        else:
            self.items = None
        if save_items and self.items is not None:
            self.items.to_json(items_path, orient="records", lines=True)
    
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
        prob = np.random.rand() if len(reference_idx) > 0 else bounded_random(0.35 + 1e-12, 1)
        if prob <= 0.35 and len(reference_idx) > 0: # 35% related node
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
        elif prob <= 0.85 and len(reference_idx) > 0: # 50% related relation
            # 60% same relation with stay triple
            # 40% with reference
            relation_prob = np.random.rand()
            if relation_prob <= 0.6 and len(stay_triple_idx) > 0:
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
    
    def build(self,
            n_reference_min=30,
            n_reference_max=50,
            stay_ratio_min=1.0,
            stay_ratio_max=1.0,
            random_state=None,
            n_pick=1):
        state = np.random.get_state()
        if random_state:
            np.random.seed(random_state)

        all_triple_idx = np.array([i for i in range(len(self.triples))]).astype(int)

        result = []
        for i, row in self.ds.iterrows():
            triple_idx = np.array(row["triple"])

            # remove arbitrary
            n_stay_min = int(np.ceil(stay_ratio_min * len(triple_idx)))
            n_stay_max = int(np.ceil(stay_ratio_max * len(triple_idx)))
            n_stay_triple = int(np.ceil(bounded_random(n_stay_min, n_stay_max)))
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
                    reference_idx = np.concatenate((reference_idx, picked_idx)).astype(int)
            
            # create link classification label
            link_cls_label = [1 if i < n_stay_triple else 0 for i in range(len(reference_idx))]
            # create node classification label
            reference_triple = self.triples[reference_idx]
            node_idx = np.concatenate((reference_triple[:,0], reference_triple[:,2]))
            node_idx = np.unique(node_idx)
            node_idx.sort()
            link_idx = np.unique(reference_triple[:,1])
            link_idx.sort()

            if len(triple_idx) > 0:
                triple = self.triples[triple_idx]
                triple_node_idx = np.concatenate((triple[:,0], triple[:,2]))
            else:
                triple_node_idx = np.array([]).astype(int)

            node_cls_label = np.isin(node_idx, triple_node_idx).astype(int).tolist()

            entry = {
                "text" : row["text"],
                "subject" : row["subject"],
                "relation" : row["relation"],
                "objects" : row["objects"],
                "reference_triple" : reference_idx,
                "reference_node" : node_idx, # ini indeks ke alias
                "reference_relation" : link_idx,
                "link_cls_label" : link_cls_label,
                "node_cls_label" : node_cls_label
            }
            result.append(entry)
        np.random.set_state(state)
        return result