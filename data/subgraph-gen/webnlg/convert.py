from datasets import load_dataset
import numpy as np
import random
import json
from tqdm import tqdm
import os

from ordered_set import OrderedSet

random.seed(42)

config = json.load(open("../config.json", 'r'))

min_ratio = config["MIN_RATIO"]
max_ratio = config["MAX_RATIO"]

def preprocess(ds, entity2id, rel2id):
    result = []
    for el1 in tqdm(ds):
        texts = el1["lex"]["text"]
        if len(texts) == 0:
            result.append({}) # corresponding empty entry
            continue
        text = random.choice(texts)
        y_triples = el1["modified_triple_sets"]["mtriple_set"][0]
        y_triples = OrderedSet([tuple(t.split(" | ")) for t in y_triples])
        x_triples = y_triples.copy()
        for el2 in ds:
            s_triples = el2["modified_triple_sets"]["mtriple_set"][0]
            s_triples = OrderedSet([tuple(t.split(" | ")) for t in s_triples])
            nodes1 = OrderedSet([t[0] for t in x_triples] + [t[1] for t in x_triples])
            nodes2 = OrderedSet([t[0] for t in s_triples] + [t[1] for t in s_triples])

            if len(nodes1.intersection(nodes2)) > 0:
                x_triples = x_triples.union(s_triples)
            
            ratio = len(y_triples) / len(x_triples)
            random_threshold = (random.random() * (max_ratio - min_ratio)) + min_ratio
            if ratio <= random_threshold:
                break
        
        all_entities = list(OrderedSet([t[0] for t in x_triples] + [t[2] for t in x_triples]))
        all_relations = list(OrderedSet([t[1] for t in x_triples]))

        y_entities = list(OrderedSet([t[0] for t in y_triples] + [t[2] for t in y_triples]))

        all_entities = sorted([entity2id[n] for n in all_entities])
        all_relations = sorted([rel2id[r] for r in all_relations])

        y_node_cls = [int(el in y_entities) for el in all_entities]

        internal_entity2id = {k : v for v, k in enumerate(all_entities)}
        internal_rel2id = {k : v for v, k in enumerate(all_relations)}

        x_coo = [[internal_entity2id[entity2id[el[0]]], internal_rel2id[rel2id[el[1]]], internal_entity2id[entity2id[el[2]]]] for el in x_triples]
        y_coo = [[internal_entity2id[entity2id[el[0]]], internal_rel2id[rel2id[el[1]]], internal_entity2id[entity2id[el[2]]]] for el in y_triples]

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


ds = load_dataset("web_nlg", 'release_v3.0_en')

len_train = len(ds["train"])
len_val = len(ds["dev"])

ds = ds["train"].to_list() + ds["dev"].to_list() + ds["test"].to_list()

entities = OrderedSet()
relations = OrderedSet()

for el in ds:
    triples = el["modified_triple_sets"]["mtriple_set"][0]
    triples = [t.split(" | ") for t in triples]
    for s, r, o in triples:
        entities.add(s)
        relations.add(r)
        entities.add(o)

if not os.path.exists("./proc"):
    os.makedirs("./proc")

entities = list(entities)
relations = list(relations)

with open("./proc/entities.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(entities))

with open("./proc/relations.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(relations))

entity2id = {el : i for i, el in enumerate(entities)}
rel2id = {el : i for i, el in enumerate(relations)}

ds = preprocess(ds, entity2id, rel2id)

train, val, test = ds[:len_train], ds[len_train:len_train+len_val], ds[len_train+len_val:]

train = [el for el in train if len(el) > 0]
val = [el for el in val if len(el) > 0]
test = [el for el in test if len(el) > 0]

with open("./proc/train.json", 'w', encoding="utf-8") as fp:
    json.dump(train, fp)

with open("./proc/dev.json", 'w', encoding="utf-8") as fp:
    json.dump(val, fp)

with open("./proc/test.json", 'w', encoding="utf-8") as fp:
    json.dump(test, fp)