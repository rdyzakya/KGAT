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

class SubgraphGenerationDataset(Dataset):
    def __init__(self, path, id2entity, id2relation, triples):
        self.data = load_json(path)
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.triples = triples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        # text, entities, relations, x_coo, y_coo/y_node_cls
        text = self.data[i]["text"]

        entities = self.data[i]["entities"]
        entities = [self.id2entity[e] for e in entities]

        relations = self.data[i]["relations"]
        relations = [self.id2relation[r] for r in relations]

        x_coo = torch.tensor(self.data[i]["x_coo"]) # transpose. 0 - subject ; 1 - relation ; 2 - object

        y_node_cls = self.data[i]["y_coo_cls"]
        y_coo_mask = [bool(el) for el in y_node_cls]
        y_coo = x_coo[y_coo_mask]

        x_coo = x_coo.T
        y_coo = y_coo.T

class LMKBCDataset(Dataset):
    pass