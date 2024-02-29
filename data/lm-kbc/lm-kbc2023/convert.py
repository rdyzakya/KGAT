import pandas as pd
import json
import os

raw_dir = "./raw"
proc_dir = "./proc"
data_split = ["train", "val", "test"]

def convert(raw_path, res_path):
    df = pd.read_json(path_or_buf=raw_path, lines=True)

    num_entities = 0
    num_relation = 0
    entity2id = dict()
    relation2id = dict()
    triples = []

    for i, row in df.iterrows():
        s, o, r = row.SubjectEntity, row.ObjectEntities, row.Relation
        o = [el for el in o if len(el) > 0]

        if s not in entity2id.keys():
            entity2id[s] = num_entities
            num_entities += 1
        
        for obj in o:
            if obj not in entity2id.keys():
                entity2id[obj] = num_entities
                num_entities += 1
        
        if r not in relation2id.keys():
            relation2id[r] = num_relation
            num_relation += 1
        
        for obj in o:
            triples.append((s, obj, r))
    
    coo = []
    for s, o, r in triples:
        coo.append([
            entity2id[s], entity2id[o], relation2id[r]
        ])
    
    id2entity = {}

    for k, v in entity2id.items():
        if v not in id2entity.keys():
            id2entity[v] = []
        id2entity[v].append(k)

    id2relation = {}

    for k, v in relation2id.items():
        id2relation[v] = k
    
    data = {
        "num_triplets" : len(coo),
        "num_entities" : num_entities,
        "num_relations" : num_relation,
        "entity" : id2entity,
        "relation" : id2relation,
        "coo" : coo
    }

    with open(res_path, 'w', encoding="utf-8") as fp:
        json.dump(data, fp)
    
    return data

if not os.path.exists(proc_dir):
    os.makedirs(proc_dir)

for split in data_split:
    raw_path = os.path.join(raw_dir, f"{split}.jsonl")
    res_path = os.path.join(proc_dir, f"{split}.json")
    convert(raw_path, res_path)