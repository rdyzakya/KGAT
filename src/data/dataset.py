from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from .prompt import Prompt
from ._data_utils import (
    read_txt
)
from utils import EMPTY_OBJECT
from disambiguation import my_disambiguation
from tqdm import tqdm
import re

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
            texts_attr = torch.load(texts_tensor_path, weights_only=True)[sentence_emb_mode]
            texts_attr = texts_attr[sentence_emb_index] if texts_attr.dim() == 3 else texts_attr
            assert texts_attr.dim() == 2
            self.texts_attr = texts_attr

        if entities_tensor_path is None:
            self.entities_attr = None
        else:
            entities_attr = torch.load(entities_tensor_path, weights_only=True)[sentence_emb_mode]
            entities_attr = entities_attr[sentence_emb_index] if entities_attr.dim() == 3 else entities_attr
            assert entities_attr.dim() == 2
            self.entities_attr = entities_attr
        
        if relations_tensor_path is None:
            self.relations_attr = None
        else:
            relations_attr = torch.load(relations_tensor_path, weights_only=True)[sentence_emb_mode]
            relations_attr = relations_attr[sentence_emb_index] if relations_attr.dim() == 3 else relations_attr
            assert relations_attr.dim() == 2
            self.relations_attr = relations_attr

        self.sentence_emb_mode = sentence_emb_mode
        self.sentence_emb_index = sentence_emb_index
    
    def prepare_train(self):
        raise NotImplementedError
    
    def prepare_eval(self):
        raise NotImplementedError
    
    def __len__(self):
        if "data" not in dir(self):
            raise ValueError("Please run prepare_train or prepare_eval method")
        return len(self.data)
    
    def __getitem__(self, idx):
        if "data" not in dir(self):
            raise ValueError("Please run prepare_train or prepare_eval method")
        return self.data[idx]
    
    def get(self, idx):
        raise NotImplementedError
    
    def collate_fn(self, batch):
        raise NotImplementedError
    
class SubgraphGenDataset(KGATDataset):
    def prepare_train(self):
        result = []
        for i, row in tqdm(self.items.iterrows()):
            for text_idx in row["text"]:
                text_idx = text_idx # injection_node
                nodes_alias_idx = row["reference_node"] # x
                triples_idx = row["reference_triple"] # edge_index
                relations_idx = row["reference_relation"] # relations
                link_cls_label = row["link_cls_label"]
                node_cls_label = row["node_cls_label"]
                obj_alias_idx = row["objects"]
                result.append((
                    # X
                    text_idx,
                    nodes_alias_idx,
                    triples_idx,
                    relations_idx,
                    # Y
                    link_cls_label,
                    node_cls_label,
                    obj_alias_idx
                ))
        self.data = result
        return result
    
    def prepare_eval(self):
        return self.prepare_train()
    
    def get(self, idx, alias_idx=None):
        ( # X
        text_idx,
        nodes_alias_idx,
        triples_idx,
        relations_idx,
        # Y
        link_cls_label,
        node_cls_label,
        obj_alias_idx) =  self.data[idx]
        
        nodes_idx = [np.random.choice(el) if alias_idx is None else el[alias_idx] for el in self.entities_alias.loc[nodes_alias_idx, "alias_idx"]]

        obj_idx = [np.random.choice(el) if alias_idx is None else el[alias_idx] for el in self.entities_alias.loc[obj_alias_idx, "alias_idx"]]

        triples = [self.triples[ti] for ti in triples_idx]

        enttity_edge_idx_mapping = {
            el : i for i, el in enumerate(nodes_alias_idx)
        }
        relation_edge_idx_mapping = {
            el : i for i, el in enumerate(relations_idx)
        }

        triples = [
            [enttity_edge_idx_mapping[el[0]], relation_edge_idx_mapping[el[1]], enttity_edge_idx_mapping[el[2]]] for el in triples
        ]

        edge_index = np.transpose(triples)
        edge_index = torch.from_numpy(edge_index)

        return {
            "x" : self.entities_attr[nodes_idx],
            "edge_index" : edge_index,
            "relations" : self.relations_attr[relations_idx],
            "query" : self.texts_attr[text_idx].unsqueeze(0),
            "values" : self.entities_attr[obj_idx] if len(obj_idx) > 0 else None,
            "node_batch" : torch.zeros(len(nodes_idx)).int(),
            "query_batch" : torch.zeros(1).int(),
            "values_batch" : torch.zeros(len(obj_idx)).int() if len(obj_idx) > 0 else None,
            "link_cls_label" : torch.tensor(link_cls_label),
            "node_cls_label" : torch.tensor(node_cls_label),
        }


class LMKBCDataset(KGATDataset):
    def __init__(self, 
                builder,
                texts_txt_path,
                entities_txt_path,
                relations_txt_path,
                entities_alias_path,
                tokenizer=None,
                n_tokens=1, 
                texts_tensor_path=None,
                entities_tensor_path=None,
                relations_tensor_path=None,
                sentence_emb_mode="eol",
                sentence_emb_index=None):
        super().__init__(
            builder,
            texts_txt_path,
            entities_txt_path,
            relations_txt_path,
            entities_alias_path,
            texts_tensor_path=texts_tensor_path,
            entities_tensor_path=entities_tensor_path,
            relations_tensor_path=relations_tensor_path,
            sentence_emb_mode=sentence_emb_mode,
            sentence_emb_index=sentence_emb_index
        )
        self.tokenizer = tokenizer
        self.n_tokens = n_tokens
        self.negative_objects = [list() for _ in range(self.items.shape[0])]
        self.prompt = Prompt()
    
    def prepare_train(self, prompt_idx=None):
        result = []
        all_prompt_idx = []
        for i, row in tqdm(self.items.iterrows()):
            text_idx = row["text"][0] # injection_node
            subject_alias_ids = self.entities_alias.loc[row["subject"],"alias_idx"]
            subject_aliases = [self.entities[sid] for sid in subject_alias_ids]
            relation = self.relations[row["relation"]]

            for s_alias in subject_aliases:
                nodes_alias_idx = row["reference_node"] # x
                triples_idx = row["reference_triple"] # edge_index
                relations_idx = row["reference_relation"] # relations
                if len(row["objects"]) == 0:

                    prompt, pid = self.prompt.pick(subject=s_alias,
                                                    relation=relation,
                                                    object=EMPTY_OBJECT,
                                                    n_tokens=self.n_tokens,
                                                    negative_sample=False,
                                                    inference=False,
                                                    idx=prompt_idx)
                    all_prompt_idx.append(pid)
                    result.append((
                        # x
                        ## GRAPH
                        text_idx,
                        nodes_alias_idx,
                        triples_idx,
                        relations_idx,
                        ## TEXT
                        prompt,
                    ))
                else:
                    for obj_idx in row["objects"]:
                        object_alias_ids = self.entities_alias.loc[obj_idx, "alias_idx"]
                        object_aliases = [self.entities[oid] for oid in object_alias_ids]
                        for o_alias in object_aliases:
                            prompt, pid = self.prompt.pick(subject=s_alias,
                                                            relation=relation,
                                                            object=o_alias,
                                                            n_tokens=self.n_tokens,
                                                            negative_sample=False,
                                                            inference=False,
                                                            idx=prompt_idx)
                            all_prompt_idx.append(pid)

                            result.append((
                                # x
                                ## GRAPH
                                text_idx,
                                nodes_alias_idx,
                                triples_idx,
                                relations_idx,
                                ## TEXT
                                prompt,
                            ))
                for n_obj in self.negative_objects[i]:
                    prompt, pid = self.prompt.pick(subject=s_alias,
                                                    relation=relation,
                                                    object=n_obj,
                                                    n_tokens=self.n_tokens,
                                                    negative_sample=True,
                                                    inference=False,
                                                    idx=prompt_idx)
                    all_prompt_idx.append(pid)
                    result.append((
                        # x
                        ## GRAPH
                        text_idx,
                        nodes_alias_idx,
                        triples_idx,
                        relations_idx,
                        ## TEXT
                        prompt,
                    ))

        self.data = result
        self.prompt_idx = all_prompt_idx
        return result, all_prompt_idx
    
    def prepare_eval(self, prompt_idx=None):
        result = []
        all_prompt_idx = []
        for i, row in tqdm(self.items.iterrows()):
            text_idx = row["text"][0] # injection_node
            nodes_alias_idx = row["reference_node"] # x
            triples_idx = row["reference_triple"] # edge_index
            relations_idx = row["reference_relation"] # relations
            subject_ids = self.entities_alias.loc[row["subject"],"alias_idx"]
            relation = self.relations[row["relation"]]

            s_alias = self.entities[subject_ids[0]]

            prompt, pid = self.prompt.pick(subject=s_alias,
                                            relation=relation,
                                            object=None,
                                            n_tokens=self.n_tokens,
                                            negative_sample=False,
                                            inference=True,
                                            idx=prompt_idx)
            all_prompt_idx.append(pid)
            result.append((
                # x
                ## GRAPH
                text_idx,
                nodes_alias_idx,
                triples_idx,
                relations_idx,
                ## TEXT
                prompt,
            ))
        self.data = result
        self.prompt_idx = all_prompt_idx
        return result, all_prompt_idx
    
    def augment(self, outputs):
        assert len(outputs) == self.items.shape[0] == len(self.prompt_idx)
        self.negative_objects = []
        for i in tqdm(range(len(outputs))):
            ground_truth_obj_ids = self.items.loc[i, "objects"]
            ground_truth_qids = []
            entry = []
            for oid in ground_truth_obj_ids:
                if re.match(r"Q\d+", self.entities_alias.loc[oid, "id"]):
                    ground_truth_qids.append(self.entities_alias.loc[oid, "id"])
                else:
                    object_alias_idx = self.entities_alias.loc[oid, "alias_idx"]
                    objects = [self.entities[oaid] for oaid in object_alias_idx]
                    for o in objects:
                        ground_truth_qids.append(my_disambiguation(o))
            for out in outputs[i]:
                obj, true_or_false = self.prompt.regex(self.prompt_idx[i], out)
                predicted_qid = my_disambiguation(obj)
                if predicted_qid not in ground_truth_qids:
                    entry.append(obj)
            self.negative_objects.append(entry)
        return self.negative_objects
    
    def get(self, idx, alias_idx=None, inference=False):
        (
            # x
            ## GRAPH
            text_idx,
            nodes_alias_idx,
            triples_idx,
            relations_idx,
            ## TEXT
            prompt,
        ) =  self.data[idx]
        
        nodes_idx = [np.random.choice(el) if alias_idx is None else el[alias_idx] for el in self.entities_alias.loc[nodes_alias_idx, "alias_idx"]]

        triples = [self.triples[ti] for ti in triples_idx]

        enttity_edge_idx_mapping = {
            el : i for i, el in enumerate(nodes_alias_idx)
        }
        relation_edge_idx_mapping = {
            el : i for i, el in enumerate(relations_idx)
        }

        triples = [
            [enttity_edge_idx_mapping[el[0]], relation_edge_idx_mapping[el[1]], enttity_edge_idx_mapping[el[2]]] for el in triples
        ]

        edge_index = np.transpose(triples)
        edge_index = torch.from_numpy(edge_index)

        # return self.texts_attr[text_idx], self.entities_attr[nodes_idx], edge_index, self.relations_attr[relations_idx], link_cls_label, node_cls_label
        tokenized = self.tokenizer(prompt, padding=True, return_tensors="pt")

        if (tokenized["input_ids"][:,-1] == self.tokenizer.eos_token_id).all().logical_not():
            tokenized["input_ids"] = torch.cat([tokenized["input_ids"], torch.full((1,1), self.tokenizer.eos_token_id)], dim=1)
            tokenized["attention_mask"] = torch.cat([tokenized["attention_mask"], torch.ones(1,1, dtype=tokenized["attention_mask"].dtype)], dim=1)

        return {
            "x" : self.entities_attr[nodes_idx],
            "edge_index" : edge_index,
            "relations" : self.relations_attr[relations_idx],
            "injection_node" : self.texts_attr[text_idx].unsqueeze(0),
            "node_batch" : torch.zeros(len(nodes_idx)).int(),
            "injection_node_batch" : torch.zeros(1).int(),
            "input_ids" : tokenized["input_ids"],
            "attention_mask" : tokenized["attention_mask"]
        }