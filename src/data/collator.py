import numpy as np
import torch

class SubgraphGenCollator:
    def __init__(self, ds, alias_idx=None):
        self.ds = ds
        self.alias_idx = alias_idx # recommend : 0
    
    def __call__(self, batch):
        (
            # x
            text_idx,
            nodes_alias_idx,
            triples_idx,
            relations_idx,
            # Y
            link_cls_label,
            node_cls_label,
        ) = zip(*batch)

        node_batch = []
        for i, el in enumerate(nodes_alias_idx):
            node_batch.append(
                torch.full((len(el), ), i)
            )
        node_batch = torch.cat(node_batch)

        nodes_alias_idx = np.concatenate(nodes_alias_idx)
        nodes_alias_idx = nodes_alias_idx.flatten() # numpy array
        nodes_idx = [np.random.choice(el) if self.alias_idx is None else el[self.alias_idx] for el in self.ds.entities_alias.loc[nodes_alias_idx, "alias_idx"]]

        relations_idx = np.concatenate(relations_idx)
        relations_idx = np.unique(relations_idx) # relation must unique

        enttity_edge_idx_mapping = {
            el : i for i, el in enumerate(nodes_alias_idx)
        }
        relation_edge_idx_mapping = {
            el : i for i, el in enumerate(relations_idx)
        }

        triples_idx = np.concatenate(triples_idx)
        triples = [self.ds.triples[ti] for ti in triples_idx]
        triples = [
            [enttity_edge_idx_mapping[el[0]], relation_edge_idx_mapping[el[1]], enttity_edge_idx_mapping[el[2]]] for el in triples
        ]

        edge_index = np.transpose(triples)
        edge_index = torch.from_numpy(edge_index)

        link_cls_label = np.concatenate(link_cls_label)
        link_cls_label = torch.from_numpy(link_cls_label)

        node_cls_label = np.concatenate(node_cls_label)
        node_cls_label = torch.from_numpy(node_cls_label)

        return {
            "x" : self.ds.entities_attr[nodes_idx],
            "edge_index" : edge_index,
            "relations" : self.ds.relations_attr[relations_idx],
            "injection_node" : self.ds.texts_attr[list(text_idx)],
            "node_batch" : node_batch,
            "injection_node_batch" : torch.arange(0, len(text_idx)),
            "link_cls_label" : link_cls_label,
            "node_cls_label" : node_cls_label
        }

class LMKBCCollator:
    def __init__(self, ds, tokenizer=None, alias_idx=None):
        self.ds = ds
        self.tokenizer = self.ds.tokenizer or tokenizer
        self.alias_idx = alias_idx # recommend : 0
    
    def __call__(self, batch):
        (
            # GRAPH
            text_idx,
            nodes_alias_idx,
            triples_idx,
            relations_idx,
            # TEXT
            prompt
        ) = zip(*batch)

        node_batch = []
        for i, el in enumerate(nodes_alias_idx):
            node_batch.append(
                torch.full((len(el), ), i)
            )
        node_batch = torch.cat(node_batch)

        nodes_alias_idx = np.concatenate(nodes_alias_idx)
        nodes_alias_idx = nodes_alias_idx.flatten() # numpy array
        nodes_idx = [np.random.choice(el) if self.alias_idx is None else el[self.alias_idx] for el in self.ds.entities_alias.loc[nodes_alias_idx, "alias_idx"]]

        relations_idx = np.concatenate(relations_idx)
        relations_idx = np.unique(relations_idx) # relation must unique

        enttity_edge_idx_mapping = {
            el : i for i, el in enumerate(nodes_alias_idx)
        }
        relation_edge_idx_mapping = {
            el : i for i, el in enumerate(relations_idx)
        }

        triples_idx = np.concatenate(triples_idx)
        triples = [self.ds.triples[ti] for ti in triples_idx]
        triples = [
            [enttity_edge_idx_mapping[el[0]], relation_edge_idx_mapping[el[1]], enttity_edge_idx_mapping[el[2]]] for el in triples
        ]

        edge_index = np.transpose(triples)
        edge_index = torch.from_numpy(edge_index)

        tokenized = self.tokenizer(prompt, padding=True, return_tensors="pt")

        if (tokenized["input_ids"][:,-1] == self.tokenizer.eos_token_id).all().logical_not():
            tokenized["input_ids"] = torch.cat([tokenized["input_ids"], torch.full((len(prompt),1), self.tokenizer.eos_token_id)], dim=1)
            tokenized["attention_mask"] = torch.cat([tokenized["attention_mask"], torch.ones(len(prompt),1, dtype=tokenized["attention_mask"].dtype)], dim=1)

        return {
            "x" : self.ds.entities_attr[nodes_idx],
            "edge_index" : edge_index,
            "relations" : self.ds.relations_attr[relations_idx],
            "injection_node" : self.ds.texts_attr[list(text_idx)],
            "node_batch" : node_batch,
            "injection_node_batch" : torch.arange(0, len(text_idx)),
            "input_ids" : tokenized["input_ids"],
            "attention_mask" : tokenized["attention_mask"]
        }