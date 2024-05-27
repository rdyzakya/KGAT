import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import os

from ordered_set import OrderedSet

random.seed(42)

data = pd.read_csv("./raw/v4_atomic_all_agg.csv")

ds = []
nodes = OrderedSet()
for i, row in data.iterrows():
    text = row["event"]
    nodes.add(text)
    triples = []

    row = row.to_dict()
    del row["event"]
    del row["prefix"]
    del row["split"]

    for k, v in row.items():
        v = list(OrderedSet(eval(v)))
        for vv in v:
            if vv == "none":
                continue
            nodes.add(vv)
            triples.append(
                (text, vv, k)
            )
    ds.append({
        "text" : text,
        "triples" : triples
    })

nodes = list(nodes)
relations = [
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xWant"
]

if not os.path.exists("./proc"):
    os.makedirs("./proc")

with open("./proc/entities.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(nodes))

with open("./proc/relations.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(relations))

entity_map = {el : i for i, el in enumerate(nodes)}
rel_map = {el : i for i, el in enumerate(relations)}

min_ratio = 0.1
max_ratio = 0.5

ds2 = []
for el in tqdm(ds):
    el_triples = OrderedSet(el["triples"])
    res_triples = el_triples.copy()
    for s in ds:
        s_triples = s["triples"]

        if el_triples == s_triples:
            continue

        nodes1 = OrderedSet([t[1] for t in res_triples])
        nodes2 = OrderedSet([t[1] for t in s_triples])

        if len(nodes1.intersection(nodes2)) > 0:
            res_triples = res_triples.union(s_triples)

        ratio = len(el_triples) / len(res_triples)
        random_threshold = (random.random() * (max_ratio - min_ratio)) + min_ratio
        if ratio <= random_threshold:
            break
    all_nodes = list(OrderedSet([t[0] for t in res_triples] + [t[1] for t in res_triples]))
    all_rels = list(OrderedSet([t[2] for t in res_triples]))

    entities = sorted([entity_map[n] for n in all_nodes])
    relations = sorted([rel_map[r] for r in all_rels])

    exist_entities = list(OrderedSet([t[0] for t in el_triples] + [t[1] for t in el_triples]))
    exist_entities = [entity_map[n] for n in exist_entities]

    y_node_cls = [int(n in exist_entities) for n in entities]

    el_triples = list(el_triples)
    res_triples = list(res_triples)

    en_map = {e : i for i, e in enumerate(entities)}
    r_map = {r : i for i, r in enumerate(relations)}

    y_coo = [
        [en_map[entity_map[t[0]]], r_map[rel_map[t[2]]], en_map[entity_map[t[1]]]] for t in el_triples
    ]

    x_coo = [
        [en_map[entity_map[t[0]]], r_map[rel_map[t[2]]], en_map[entity_map[t[1]]]] for t in res_triples
    ]

    y_coo_cls = [int(el in y_coo) for el in x_coo]
    
    ds2.append({
        "text" : el["text"],
        "entities" : entities,
        "relations" : relations,
        "x_coo" : x_coo,
        "y_coo_cls" : y_coo_cls,
        "y_node_cls" : y_node_cls
    })

random.shuffle(ds2)

train_ratio = 0.8
val_ratio = 0.1

train_ds = ds2[:int(train_ratio * len(ds2))]
dev_ds = ds2[len(train_ds):int(val_ratio * len(ds2)) + len(train_ds)]
test_ds = ds2[int(val_ratio * len(ds2)) + len(train_ds):]

with open("./proc/train.json", 'w', encoding="utf-8") as fp:
    json.dump(train_ds, fp)

with open("./proc/dev.json", 'w', encoding="utf-8") as fp:
    json.dump(dev_ds, fp)

with open("./proc/test.json", 'w', encoding="utf-8") as fp:
    json.dump(test_ds, fp)