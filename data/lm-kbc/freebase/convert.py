import json
import pandas as pd
import random
import numpy as np
from ordered_set import OrderedSet
from tqdm import tqdm
import os

random.seed(42)
np.random.seed(42)

N = 500

def find_reference(sub_id, rel_id, obj_ids, df):
    n_triples = random.randint(20,500)
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

def load_dataset(path):
    with open(path, 'r', encoding="utf-8") as fp:
        data = fp.read().strip().splitlines()
    data = [{ i : v for i, v in enumerate(el.split())} for el in data]
    data = pd.DataFrame(data)
    return data

def decode(ds_, mid2name):
    ds = ds_.copy()
    ds[0] = ds[0].apply(mid2name.get)
    ds[1] = ds[1].apply(lambda x: x.split('/')[-1])
    ds[2] = ds[2].apply(mid2name.get)
    return ds

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

train_path = "./raw/Release/train.txt"
val_path = "./raw/Release/valid.txt"
test_path = "./raw/Release/test.txt"

mid2name_path = "./raw/mid2name.tsv"

train = load_dataset(train_path)
val = load_dataset(val_path)
test = load_dataset(test_path)

mid2name = pd.read_csv(mid2name_path, sep='\t', header=None)
mid2name.index = mid2name[0]
mid2name = mid2name.to_dict()[1]

train = decode(train, mid2name)
val = decode(val, mid2name)
test = decode(test, mid2name)

df = pd.concat([train, val, test])
df = df.dropna().reset_index(drop=True)

entities = list(set(df[0].values.tolist() + df[2].values.tolist()))
entity2id = {el : i for i, el in enumerate(entities)}

relations = df[1].unique().tolist()
relation2id = {el : i for i, el in enumerate(relations)}

df = df[[0,1,2]] # subject - relation - object
df[0] = df[0].apply(entity2id.get)
df[1] = df[1].apply(relation2id.get)
df[2] = df[2].apply(entity2id.get)

triples = [list(el.values()) for el in df.to_dict(orient="records")]

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