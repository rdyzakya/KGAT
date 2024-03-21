import pandas as pd
import random
from tqdm import tqdm
import numpy as np
import json
import os

random.seed(42)

min_ratio = 0.1
max_ratio = 0.5

def preprocess1(df, rel):
    result = []
    for i, row in df.iterrows():
        entities = row[1].split(" ; ")
        masks = row[2].split()
        assert len(entities) == len(masks)

        triples = row[3].split(" ; ")
        for i, t in enumerate(triples):
            s, r, o = t.split()
            s, r, o = int(s), int(r), int(o)
            triples[i] = (entities[s],entities[o],rel[r]) # subject - object - relation
        
        masks = [el[:-1] + f"_{i}>" for i, el in enumerate(masks)]
        text = row[4]
        for i, m in enumerate(masks):
            text = text.replace(m, entities[i])
        result.append({
            "text" : text,
            "entities" : entities,
            "triples" : triples
        })
    return result

def preprocess2(ds, all_ds, entity2id, rel2id):
    result = []
    for el in tqdm(ds):
        text = el["text"]
        entities = el["entities"]
        all_entities = entities.copy()
        y_triples = set(el["triples"])
        x_triples = y_triples.copy()
        for s in all_ds:
            s_triples = set(s["triples"])
            s_entities = s["entities"]
            if y_triples == s_triples:
                continue

            nodes1 = set([t[0] for t in x_triples] + [t[1] for t in x_triples])
            nodes2 = set([t[0] for t in s_triples] + [t[1] for t in s_triples])

            if len(nodes1.intersection(nodes2)) > 0:
                x_triples = x_triples.union(s_triples)
                all_entities.extend(s_entities)
            
            ratio = len(y_triples) / len(x_triples)
            random_threshold = (random.random() * (max_ratio - min_ratio)) + min_ratio
            if ratio <= random_threshold:
                break
        
        # still in the form of string
        x_triples = list(x_triples)
        y_triples = list(y_triples)
        
        all_relations = list(set([t[2] for t in x_triples]))

        all_entities = sorted([entity2id[el] for el in all_entities])
        all_relations = sorted([rel2id[el] for el in all_relations])
        entities = [entity2id[el] for el in entities]

        y_node_cls = [int(el in entities) for el in all_entities]

        internal_entity2id = {k : v for v, k in enumerate(all_entities)}
        internal_rel2id = {k : v for v, k in enumerate(all_relations)}

        x_coo = [[internal_entity2id[entity2id[el[0]]], internal_rel2id[rel2id[el[2]]], internal_entity2id[entity2id[el[1]]]] for el in x_triples]
        y_coo = [[internal_entity2id[entity2id[el[0]]], internal_rel2id[rel2id[el[2]]], internal_entity2id[entity2id[el[1]]]] for el in y_triples]

        y_coo_cls = [int(el in y_coo) for el in x_coo]

        result.append({
            "text" : text,
            "entities" : all_entities,
            "relations" : all_relations,
            "x_coo" : x_coo,
            "y_coo_cls" : y_coo_cls,
            "y_node_cls" : y_node_cls
        })
    
    return result

train_path = "./raw/preprocessed.train.tsv"
val_path = "./raw/preprocessed.val.tsv"
test_path = "./raw/preprocessed.test.tsv"

rel_path = "./raw/relations.vocab"

train = pd.read_csv(train_path, sep='\t', header=None)
val = pd.read_csv(val_path, sep='\t', header=None)
test = pd.read_csv(test_path, sep='\t', header=None)

with open(rel_path, 'r') as fp:
    relations = fp.read().strip().splitlines()

train = preprocess1(train, relations)
val = preprocess1(val, relations)
test = preprocess1(test, relations)

all_ds = train + val + test

entities = []
for el in all_ds:
    entities.extend(el["entities"])
entities = list(set(entities))

if not os.path.exists("./proc"):
    os.makedirs("./proc")

with open("./proc/entities.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(entities))
with open("./proc/relations.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(relations))

entity2id = {el : i for i, el in enumerate(entities)}
rel2id = {el : i for i, el in enumerate(relations)}

train = preprocess2(train, all_ds, entity2id, rel2id)
val = preprocess2(val, all_ds, entity2id, rel2id)
test = preprocess2(test, all_ds, entity2id, rel2id)

with open("./proc/train.json", 'w') as fp:
    json.dump(train, fp)

with open("./proc/val.json", 'w') as fp:
    json.dump(val, fp)

with open("./proc/test.json", 'w') as fp:
    json.dump(test, fp)