from torch.utils.data import Dataset
from ..utils import Mask
import torch
import json

# def apply_template(text, subject, relation, objects=None):
#     text = text.replace(Mask.SUBJECT_MASK, subject).replace(Mask.RELATION_MASK, relation)
#     if objects is None:
#         return text
#     return text.replace(Mask.OBJECT_MASK, str(objects))

def load_json(path):
    with open(path, 'r', encoding="utf-8") as fp:
        data = json.load(fp)
    return data

def load_id2map(path):
    with open(path, 'r', encoding="utf-8") as fp:
        data = fp.read().strip().splitlines()
    data = {i : el for i, el in enumerate(data)}
    return data

class SubgraphGenerationDataset(Dataset):
    def __init__(self, path, id2entity, id2relation, n_data=None):
        self.data = load_json(path)
        self.data = self.data if not n_data else self.data[:n_data]
        self.id2entity = id2entity
        self.id2relation = id2relation
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        # text, entities, relations, x_coo, y_coo/y_node_cls
        text = self.data[i]["text"]

        entities = self.data[i]["entities"]
        entities = [self.id2entity[e] for e in entities]

        relations = self.data[i]["relations"]
        relations = [self.id2relation[r] for r in relations]

        x_coo = torch.tensor(self.data[i]["x_coo"]) # 0 - subject ; 1 - relation ; 2 - object

        y_coo_cls = torch.tensor(self.data[i]["y_coo_cls"])
        # y_coo_mask = [bool(el) for el in y_coo_cls]
        # y_coo = x_coo[y_coo_mask]

        x_coo = x_coo.T
        # y_coo = y_coo.T

        return text, entities, relations, x_coo, y_coo_cls

class LMKBCDataset(Dataset):
    def __init__(self, 
                 path, 
                 id2entity, 
                 id2relation, 
                 triples, 
                 prompt_template=f"{Mask.KG_MASK} -> S : {Mask.SUBJECT_MASK} | R : {Mask.RELATION_MASK} | O : {Mask.OBJECT_MASK}", 
                 graph_query_template=f"S : {Mask.SUBJECT_MASK} | R : {Mask.RELATION_MASK}", 
                 n_data=None):
        self.data = load_json(path)
        self.data = self.data if not n_data else self.data[:n_data]
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.triples = triples
        self.prompt_template = prompt_template
        self.graph_query_template = graph_query_template
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        # text_in, entities, relations, x_coo, text_out
        subject_id = self.data[i]["subject"]
        relation_id = self.data[i]["relation"]
        object_ids = self.data[i]["objects"]
        reference = self.data[i]["reference"]

        subject = self.id2entity[subject_id]
        relation = self.id2relation[relation_id]
        objects = [self.id2entity[el] for el in object_ids]

        sr_graph_query = (self.graph_query_template
                          .replace(Mask.SUBJECT_MASK, subject)
                          .replace(Mask.RELATION_MASK, relation))
        sro_texts = []
        for o in objects:
            sro_texts.append(
                self.prompt_template
                .replace(Mask.SUBJECT_MASK, subject)
                .replace(Mask.RELATION_MASK, relation)
                .replace(Mask.OBJECT_MASK, o)
            )

        x_coo = [self.triples[el] for el in reference]
        
        entities = set()
        relations = set()

        for s, r, o in x_coo:
            entities.add(s)
            relations.add(r)
            entities.add(o)
        
        entities = list(entities)
        relations = list(relations)
        
        
        x_coo = torch.tensor([[entities.index(t[0]), relations.index(t[1]), entities.index(t[2])] for t in x_coo])  # 0 - subject ; 1 - relation ; 2 - object

        entities = [self.id2entity[el] for el in entities]
        relations = [self.id2relation[el] for el in relations]

        x_coo = x_coo.T

        # text, entities, relations, x_coo, y_coo_cls

        return sr_graph_query, entities, relations, x_coo, sro_texts

    # def process_prompt(self, subject, relation, objects):
    #     # return text_in, text_out
    #     text_in = apply_template(self.prompt_template, subject=subject, relation=relation, objects='')
    #     text_out = apply_template(self.prompt_template, subject=subject, relation=relation, objects=objects)
    #     return text_in, text_out
    
    # def process_graph_query(self, subject, relation):
    #     return apply_template(self.graph_query_template, subject=subject, relation=relation, objects=None)