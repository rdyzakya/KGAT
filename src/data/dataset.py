from torch.utils.data import Dataset
import pandas as pd
import torch
from ._data_utils import (
    read_txt
)
from utils import (
    KG_MASK,
    SUBJECT_MASK,
    RELATION_MASK,
    OBJECT_MASK,
    EMPTY_OBJECT
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
    
    def prepare_train(self):
        raise NotImplementedError
    
    def prepare_eval(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def collate_fn(self, batch):
        raise NotImplementedError
    
class SubgraphGenDataset(KGATDataset):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def prepare_train(self):
        result = []
        for i, row in self.items.iterrows():
            for text_idx in row["text"]:
                text_idx = text_idx # injection_node
                nodes_alias_idx = row["reference_node"] # x
                triples_idx = row["reference_triple"] # edge_index
                relations_idx = row["reference_relation"] # relations
                link_cls_label = row["link_cls_label"]
                node_cls_label = row["node_cls_label"]
                result.append((
                    # X
                    text_idx,
                    nodes_alias_idx,
                    triples_idx,
                    relations_idx,
                    # Y
                    link_cls_label,
                    node_cls_label
                ))
        self.data = result
        return result
    
    def prepare_eval(self):
        return self.prepare_train()


def default_lmkbc_prompt(subject, relation, object, eos_token, negative_sample=False, n_tokens=1, inference=False):
    inference_template = f'Based on the knowledge graph "{KG_MASK*n_tokens}" complete the 
    following triple with the format ( SUBJECT | RELATION | OBJECT | T/F ) and stop 
    after close bracket, fill OBJECT with {EMPTY_OBJECT} if nothing satisfy, 
    fill T/F with TRUE if you think the triple is true else fill with FALSE : ( {subject} | {relation} | '
    if inference:
        return inference
    if negative_sample:
        return f'{inference_template}{object} | FALSE ){eos_token}'
    return f'{inference_template}{object} | TRUE ){eos_token}'

class LMKBCDataset(KGATDataset):
    def __init__(self, tokenizer, n_tokens=1, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.n_tokens = n_tokens
        self.negative_objects = [list() for _ in range(self.items.shape[0])]
    
    def prepare_train(self):
        result = []
        for i, row in self.items.iterrows():
            for text_idx in row["text"]:
                text_idx = text_idx # injection_node
                subject_ids = self.entities_alias.loc[row["subject"],"alias_idx"]
                relation = self.relations[row["relation"]]

                for sid in subject_ids:
                    subject = self.entities[sid]

                    batch_entry_per_subject_relation = []
                    if len(row["objects"]) == 0:
                        pass
                    for obj_idx in row["objects"]:
                        object_ids = self.entities_alias.loc[obj_idx, "alias_idx"]
                        for oid in object_ids:
                            nodes_alias_idx = row["reference_node"] # x
                            triples_idx = row["reference_triple"] # edge_index
                            relations_idx = row["reference_relation"] # relations

                            batch_entry_per_subject_relation.append((
                                # GRAPH
                                text_idx,
                                nodes_alias_idx,
                                triples_idx,
                                relations_idx,
                                # TEXT
                            ))
        self.data = result
        return result