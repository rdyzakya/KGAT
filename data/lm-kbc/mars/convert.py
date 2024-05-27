import random
import os
import json
import random
import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

config = json.load(open("../config.json"))

N = config["N"]
N_REF = config["N_REF"]

entity_map_path = "./raw/entity2text.txt"
relation_map_path = "./raw/relation2text.txt"
triples_path = "./raw/wiki_tuple_ids.txt"

def find_reference(sub_id, rel_id, obj_ids, df):
    n_triples = random.randint(N_REF//20,N_REF)
    res = OrderedSet()
    current_idx = df.loc[(df[0] == sub_id) & (df[1] == rel_id) & (df[2].isin(obj_ids))].index
    symmetry_idx = df.loc[(df[2] == sub_id) & (df[1] == rel_id) & (df[0].isin(obj_ids))].index

    current_triple_idx = current_idx.to_list() + symmetry_idx.to_list()
    while len(res) < n_triples:
        decider = random.random()
        if decider <= 0.45: # same entity
            same_sub = df.loc[(df[0] == sub_id) | (df[2] == sub_id)]
            same_sub = same_sub.loc[~same_sub.index.isin(current_triple_idx)]
            same_sub_idx = same_sub.index.to_list()
            res = res.union(same_sub_idx)
        elif decider <= 0.5: # same relation, so many
            same_rel = df.loc[df[1] == rel_id]
            n_sample = min(random.randint(1, n_triples - len(res)), same_rel.shape[0])
            same_rel = same_rel.sample(n_sample, random_state=42)
            same_rel = same_rel.loc[~same_rel.index.isin(current_triple_idx)]
            same_rel_idx = same_rel.index.to_list()
            res = res.union(same_rel_idx)
        elif decider <= 0.75: # tailing
            if len(res) == 0:
                continue
            res_triples = df.loc[list(res)]

            res_nodes = res_triples[0].values.tolist() + res_triples[2].values.tolist()

            k = random.randint(1,len(res_nodes))
            chosen_nodes = random.choices(res_nodes, k=k)

            related_triples = df.loc[df[0].isin(chosen_nodes) | df[2].isin(chosen_nodes)]
            related_triples = related_triples.loc[~related_triples.index.isin(current_triple_idx)]
            related_triples_idx = related_triples.index.to_list()
            res = res.union(related_triples_idx)
        else: #random
            n_random_triples = random.randint(1, n_triples - len(res))
            samples = df.sample(n_random_triples, random_state=42)
            samples = samples.loc[~samples.index.isin(current_triple_idx)]
            samples_idx = samples.index.to_list()
            res = res.union(samples_idx)
    return sorted(list(res))

def load_data(path):
    with open(path, 'r', encoding="utf-8") as fp:
        data = fp.read().strip().splitlines()
    return data

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

entity_map = load_data(entity_map_path)
entity_map = [el.split('\t') for el in entity_map]
entity_map = {el[0] : el[1] for el in entity_map}

relation_map = load_data(relation_map_path)
relation_map = [el.split('\t') for el in relation_map]
relation_map = {el[0] : el[1] for el in relation_map}

entities = list(entity_map.values())
relations = list(relation_map.values())

entity2id = {el : i for i, el in enumerate(entity_map.keys())} # the id
rel2id = {el : i for i, el in enumerate(relation_map.keys())} # the id

triples = load_data(triples_path)
triples = [el.split('\t') for el in triples] # subject - relation - object

df = pd.DataFrame(triples)

df[0] = df[0].apply(entity2id.get)
df[1] = df[1].apply(rel2id.get)
df[2] = df[2].apply(entity2id.get)

df["done"] = False

ds = []

# not empty
print(f"Processing {df.shape[0]} iterations...")
for i, row in tqdm(df.sample(N, random_state=42).iterrows()):
    if row["done"]:
        continue
    sub_rel = df.loc[(df[0] == row[0]) & (df[1] == row[1])]
    df.loc[sub_rel.index, "done"] = True
    
    objects = sub_rel[2].values.tolist()

    reference = find_reference(row[0], row[1], objects, df)

    ds.append({
        "subject" : row[0],
        "relation" : row[1],
        "objects" : objects,
        "reference" : reference
    })

# empty, 15% from the not empty
print(f"Processing {int(0.15 * len(ds))} iterations...")
for i in tqdm(range(int(0.15 * len(ds)))):
    random_subject = random.choice([df[0].sample(random_state=42).iloc[0], df[2].sample(random_state=42).iloc[0]])
    related_relations = df.loc[df[0] == random_subject, 1].values.tolist()
    random_relation = df.loc[~df[1].isin(related_relations), 1].sample(random_state=42).iloc[0]

    objects = []

    reference = find_reference(random_subject, random_relation, objects, df)

    ds.append({
        "subject" : random_subject,
        "relation" : random_relation,
        "objects" : objects,
        "reference" : reference
    })

random.shuffle(ds)

train, val, test = ds[:int(len(ds) * 0.8)], ds[int(len(ds) * 0.8):int(len(ds) * 0.9)], ds[int(len(ds) * 0.9):]

if not os.path.exists("./proc"):
    os.makedirs("./proc")

with open("./proc/entities.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(entities))

with open("./proc/relations.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(relations))

with open("./proc/triples.json", 'w') as fp:
    json.dump(triples, fp, cls=NpEncoder)

with open("./proc/train.json", 'w') as fp:
    json.dump(train, fp, cls=NpEncoder)

with open("./proc/val.json", 'w') as fp:
    json.dump(val, fp, cls=NpEncoder)

with open("./proc/test.json", 'w') as fp:
    json.dump(test, fp, cls=NpEncoder)