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
            obj_alias_idx
        ) = zip(*batch)

        node_batch = []
        for i, el in enumerate(nodes_alias_idx):
            node_batch.append(
                torch.full((len(el), ), i)
            )
        node_batch = torch.cat(node_batch)

        triple_batch = []
        for i, el in enumerate(triples_idx):
            triple_batch.append(
                torch.full((len(el), ), i)
            )
        triple_batch = torch.cat(triple_batch)

        nodes_alias_idx = np.concatenate(nodes_alias_idx)
        nodes_alias_idx = nodes_alias_idx.flatten() # numpy array
        nodes_idx = [np.random.choice(el) if self.alias_idx is None else el[self.alias_idx] for el in self.ds.entities_alias.loc[nodes_alias_idx, "alias_idx"]]

        relations_idx = np.concatenate(relations_idx)
        relations_idx = np.unique(relations_idx) # relation must unique


        relation_edge_idx_mapping = {
            el : i for i, el in enumerate(relations_idx)
        }

        triples_idx = np.concatenate(triples_idx)
        triples = []
        for i, ti in enumerate(triples_idx):
            s, r, o = self.ds.triples[ti]
            si, ri, oi = None, relation_edge_idx_mapping[r], None
            # found_s = False
            # found_o = False
            for j, nai in enumerate(nodes_alias_idx):
                if s == nai and node_batch[j] == triple_batch[i] and si is None:
                    si = j
                    # found_s = True
                if o == nai and node_batch[j] == triple_batch[i] and oi is None:
                    oi = j
                    # found_o = True
                if si is not None and oi is not None:
                    break
            if si is None or oi is None:
                raise Exception("Not found!")
            triples.append(
                [si, ri, oi]
            )

        edge_index = np.transpose(triples)
        edge_index = torch.from_numpy(edge_index)

        link_cls_label = np.concatenate(link_cls_label)
        link_cls_label = torch.from_numpy(link_cls_label)

        node_cls_label = np.concatenate(node_cls_label)
        node_cls_label = torch.from_numpy(node_cls_label)

        src_batch = node_batch[edge_index[0]]
        tgt_batch = node_batch[edge_index[2]]

        if (src_batch != tgt_batch).any():
            raise ValueError(f"Intersection between batch, there are connection between different graph \n src_batch : {src_batch} \n tgt_batch : {tgt_batch}")
        
        
        objects_batch = []
        for i, el in enumerate(obj_alias_idx):
            objects_batch.append(
                torch.full((len(el), ), i)
            )
        objects_batch = torch.cat(objects_batch) if len(objects_batch) > 0 else torch.tensor([])

        obj_alias_idx = np.concatenate(obj_alias_idx)
        obj_alias_idx = obj_alias_idx.flatten() # numpy array
        obj_idx = [np.random.choice(el) if self.alias_idx is None else el[self.alias_idx] for el in self.ds.entities_alias.loc[obj_alias_idx, "alias_idx"]]


        return {
            "x" : self.ds.entities_attr[nodes_idx],
            "edge_index" : edge_index,
            "relations" : self.ds.relations_attr[relations_idx],
            "query" : self.ds.texts_attr[list(text_idx)],
            "values" : self.ds.entities_attr[obj_idx] if len(obj_idx) > 0 else None,
            "node_batch" : node_batch,
            "query_batch" : torch.arange(0, len(text_idx)),
            "values_batch" : objects_batch if len(obj_idx) > 0 else None,
            "link_cls_label" : link_cls_label,
            "node_cls_label" : node_cls_label,
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
            prompt,
            # WEIGHT
            weight
        ) = zip(*batch)

        node_batch = []
        for i, el in enumerate(nodes_alias_idx):
            node_batch.append(
                torch.full((len(el), ), i)
            )
        node_batch = torch.cat(node_batch)

        triple_batch = []
        for i, el in enumerate(triples_idx):
            triple_batch.append(
                torch.full((len(el), ), i)
            )
        triple_batch = torch.cat(triple_batch)

        nodes_alias_idx = np.concatenate(nodes_alias_idx)
        nodes_alias_idx = nodes_alias_idx.flatten() # numpy array
        nodes_idx = [np.random.choice(el) if self.alias_idx is None else el[self.alias_idx] for el in self.ds.entities_alias.loc[nodes_alias_idx, "alias_idx"]]

        relations_idx = np.concatenate(relations_idx)
        relations_idx = np.unique(relations_idx) # relation must unique

        relation_edge_idx_mapping = {
            el : i for i, el in enumerate(relations_idx)
        }

        triples_idx = np.concatenate(triples_idx)
        triples = []
        for i, ti in enumerate(triples_idx):
            s, r, o = self.ds.triples[ti]
            si, ri, oi = None, relation_edge_idx_mapping[r], None
            # found_s = False
            # found_o = False
            for j, nai in enumerate(nodes_alias_idx):
                if s == nai and node_batch[j] == triple_batch[i] and si is None:
                    si = j
                    # found_s = True
                if o == nai and node_batch[j] == triple_batch[i] and oi is None:
                    oi = j
                    # found_o = True
                if si is not None and oi is not None:
                    break
            if si is None or oi is None:
                raise Exception("Not found!")
            triples.append(
                [si, ri, oi]
            )

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
            "query" : self.ds.texts_attr[list(text_idx)],
            "node_batch" : node_batch,
            "query_batch" : torch.arange(0, len(text_idx)),
            "input_ids" : tokenized["input_ids"],
            "attention_mask" : tokenized["attention_mask"],
            "weights" : torch.tensor(weight).float()
        }