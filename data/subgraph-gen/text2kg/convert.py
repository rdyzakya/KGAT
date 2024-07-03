import pandas as pd
import os
import random
import numpy as np
import json
from tqdm import tqdm

from ordered_set import OrderedSet

random.seed(42)

dbpedia_train_dir_path = "./raw/dbpedia_webnlg/train"
dbpedia_test_dir_path = "./raw/dbpedia_webnlg/ground_truth"
wikidata_test_dir_path = "./raw/wikidata_tekgen/ground_truth"

df = pd.DataFrame()

for fname in os.listdir(dbpedia_train_dir_path):
    path = os.path.join(dbpedia_train_dir_path, fname)
    df = pd.concat([df, pd.read_json(path, lines=True)])

for fname in os.listdir(dbpedia_test_dir_path):
    path = os.path.join(dbpedia_test_dir_path, fname)
    df = pd.concat([df, pd.read_json(path, lines=True)])

for fname in os.listdir(wikidata_test_dir_path):
    path = os.path.join(wikidata_test_dir_path, fname)
    df = pd.concat([df, pd.read_json(path, lines=True)])

entities = OrderedSet()
relations = OrderedSet()

for i, row in df.iterrows():
    triples = row["triples"]

    for t in triples:
        entities.add(t["sub"])
        entities.add(t["obj"])
        relations.add(t["rel"])

entities = list(entities)
relations = list(relations)

entity2id = {el : i for i, el in enumerate(entities)}
rel2id = {el : i for i, el in enumerate(relations)}

if not os.path.exists("./proc"):
    os.makedirs("./proc")

with open("./proc/entities.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(entities))

with open("./proc/relations.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(relations))

ds = []

config = json.load(open("../config.json", 'r'))

min_ratio = config["MIN_RATIO"]
max_ratio = config["MAX_RATIO"]

for i1, row1 in tqdm(df.iterrows()):
    text = row1["sent"]
    y_triples = OrderedSet(map(lambda x: tuple((x["sub"], x["obj"], x["rel"])),row1["triples"]))
    x_triples = y_triples.copy()
    if len(x_triples) == 0:
        continue
    for i2, row2 in df.iterrows():
        if i1 == i2:
            pass
        s_triples = OrderedSet(map(lambda x: tuple((x["sub"], x["obj"], x["rel"])),row2["triples"]))

        nodes1 = OrderedSet([t[0] for t in x_triples] + [t[1] for t in x_triples])
        nodes2 = OrderedSet([t[0] for t in s_triples] + [t[1] for t in s_triples])

        if len(nodes1.intersection(nodes2)) > 0:
            x_triples = x_triples.union(s_triples)
        
        ratio = len(y_triples) / len(x_triples)
        random_threshold = (random.random() * (max_ratio - min_ratio)) + min_ratio
        if ratio <= random_threshold:
            break
    
    all_entities = list(OrderedSet([t[0] for t in x_triples] + [t[1] for t in x_triples]))
    all_relations = list(OrderedSet([t[2] for t in x_triples]))

    y_entities = list(OrderedSet([t[0] for t in y_triples] + [t[1] for t in y_triples]))

    all_entities = sorted([entity2id[n] for n in all_entities])
    all_relations = sorted([rel2id[r] for r in all_relations])

    y_node_cls = [int(el in y_entities) for el in all_entities]

    internal_entity2id = {k : v for v, k in enumerate(all_entities)}
    internal_rel2id = {k : v for v, k in enumerate(all_relations)}

    x_coo = [[internal_entity2id[entity2id[el[0]]], internal_rel2id[rel2id[el[2]]], internal_entity2id[entity2id[el[1]]]] for el in x_triples]
    y_coo = [[internal_entity2id[entity2id[el[0]]], internal_rel2id[rel2id[el[2]]], internal_entity2id[entity2id[el[1]]]] for el in y_triples]

    y_coo_cls = [int(el in y_coo) for el in x_coo]

    ds.append({
            "text" : text,
            "entities" : all_entities,
            "relations" : all_relations,
            "x_coo" : x_coo,
            "y_coo_cls" : y_coo_cls,
            "y_node_cls" : y_node_cls
        })

random.shuffle(ds)

train, val, test = ds[:int(0.8*len(ds))], ds[int(0.8*len(ds)):int(0.9*len(ds))], ds[int(0.9*len(ds)):]

with open("./proc/train.json", 'w', encoding="utf-8") as fp:
    json.dump(train, fp)

with open("./proc/val.json", 'w', encoding="utf-8") as fp:
    json.dump(val, fp)

with open("./proc/test.json", 'w', encoding="utf-8") as fp:
    json.dump(test, fp)