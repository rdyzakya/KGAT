from torch.utils.data import Dataset
import torch
import json

def load_json(path):
    with open(path, 'r', encoding="utf-8") as fp:
        data = json.load(fp)
    return data

def load_id2map(path):
    with open(path, 'r', encoding="utf-8") as fp:
        data = fp.read().strip().splitlines()
    data = {i : el for i, el in enumerate(data)}
    return data

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def subgraphgen_collate_fn(batch):
    text, entities, relations, x_coo, y_coo_cls = zip(*batch)

    text = list(text)

    batch = []

    for i, e in enumerate(entities):
        batch.extend([i for _ in range(len(e))])
    
    prev_len_entity = 0
    prev_len_relation = 0
    for i in range(len(x_coo)):
        x_coo[i][0] += prev_len_entity
        x_coo[i][1] += prev_len_relation
        x_coo[i][2] += prev_len_entity
        
        prev_len_entity += len(entities[i])
        prev_len_relation += len(relations[i])
    
    entities = flatten(entities)
    relations = flatten(relations)
    
    x_coo = torch.hstack(x_coo)
    batch = torch.tensor(batch)
    y_coo_cls = torch.hstack(y_coo_cls)
    
    return text, entities, relations, x_coo, batch, y_coo_cls

def lmkbc_collate_fn(batch):
    text_in, entities, relations, x_coo, text_out = zip(*batch)

    text_in = list(text_in)

    batch = []

    for i, e in enumerate(entities):
        batch.extend([i for _ in range(len(e))])

    prev_len_entity = 0
    prev_len_relation = 0
    for i in range(len(x_coo)):
        x_coo[i][0] += prev_len_entity
        x_coo[i][1] += prev_len_relation
        x_coo[i][2] += prev_len_entity
        
        prev_len_entity += len(entities[i])
        prev_len_relation += len(relations[i])
    
    entities = flatten(entities)
    relations = flatten(relations)
    
    x_coo = torch.hstack(x_coo)
    batch = torch.tensor(batch)

    text_out = list(text_out)

    return text_in, entities, relations, x_coo, batch, text_out

class SubgraphGenerationDataset(Dataset):
    def __init__(self, path, id2entity, id2relation):
        self.data = load_json(path)
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
    def __init__(self, path, id2entity, id2relation, triples, prompt):
        self.data = load_json(path)
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.triples = triples
        self.prompt = prompt
    
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

        text_in, text_out = self.process_input(subject, relation, objects)

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

        return text_in, entities, relations, x_coo, text_out

    def process_input(self, subject, relation, objects):
        # return text_in, text_out
        return f"S : {subject} | R : {relation} | O : ", str(objects)