import os
import json
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from ordered_set import OrderedSet

random.seed(42)
np.random.seed(42)

config = json.load(open("../config.json"))

N = config["N"]
N_REF = config["N_REF"]

def find_reference(sub_id, rel_id, obj_ids, df):
    n_triples = random.randint(N_REF//20,N_REF)
    res = OrderedSet()
    current_idx = df.loc[(df[1] == sub_id) & (df[0] == rel_id) & (df[2].isin(obj_ids))].index
    symmetry_idx = df.loc[(df[2] == sub_id) & (df[0] == rel_id) & (df[1].isin(obj_ids))].index

    current_triple_idx = current_idx.to_list() + symmetry_idx.to_list()
    while len(res) < n_triples:
        decider = random.random()
        if decider <= 0.45: # same entity
            same_sub = df.loc[(df[1] == sub_id) | (df[2] == sub_id)]
            same_sub = same_sub.loc[~same_sub.index.isin(current_triple_idx)]
            same_sub_idx = same_sub.index.to_list()
            res = res.union(same_sub_idx)
        elif decider <= 0.5: # same relation, so many
            same_rel = df.loc[df[0] == rel_id]
            n_sample = min(random.randint(1, n_triples - len(res)), same_rel.shape[0])
            same_rel = same_rel.sample(n_sample, random_state=42)
            same_rel = same_rel.loc[~same_rel.index.isin(current_triple_idx)]
            same_rel_idx = same_rel.index.to_list()
            res = res.union(same_rel_idx)
        elif decider <= 0.75: # tailing
            if len(res) == 0:
                continue
            res_triples = df.loc[list(res)]

            res_nodes = res_triples[1].values.tolist() + res_triples[2].values.tolist()

            k = random.randint(1,len(res_nodes))
            chosen_nodes = random.choices(res_nodes, k=k)

            related_triples = df.loc[df[1].isin(chosen_nodes) | df[2].isin(chosen_nodes)]
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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

path = "./raw/data_preprocessed_release/cpnet/conceptnet.en.csv"

df = pd.read_csv(path, sep="\t", header=None)
df = df.fillna(value="nan")

entities = list(OrderedSet(df[1].values.tolist() + df[2].values.tolist()))
entity2id = {el : i for i, el in enumerate(entities)}

relations = df[0].unique().tolist()
relation2id = {el : i for i, el in enumerate(relations)}

df_copy = df.copy()
df_copy[0] = df_copy[0].apply(relation2id.get)
df_copy[1] = df_copy[1].apply(entity2id.get)
df_copy[2] = df_copy[2].apply(entity2id.get)

df_copy = df_copy[[1,0,2]] # subject - relation - object

triples = [list(el.values()) for el in df_copy.to_dict(orient="records")]

df_copy["done"] = False
ds = []

# not empty
print(f"Processing {df_copy.shape[0]} iterations...")
for i, row in tqdm(df_copy.sample(N, random_state=42).iterrows()):
    if row["done"]:
        continue
    sub_rel = df_copy.loc[(df_copy[0] == row[0]) & (df_copy[1] == row[1])]
    df_copy.loc[sub_rel.index, "done"] = True
    
    objects = sub_rel[2].values.tolist()

    reference = find_reference(row[1], row[0], objects, df_copy)

    ds.append({
        "subject" : row[1],
        "relation" : row[0],
        "objects" : objects,
        "reference" : reference
    })

# empty, 15% from the not empty
print(f"Processing {int(0.15 * len(ds))} iterations...")
for i in tqdm(range(int(0.15 * len(ds)))):
    random_subject = random.choice([df_copy[1].sample(random_state=42).iloc[0], df_copy[2].sample(random_state=42).iloc[0]])
    related_relations = df_copy.loc[df_copy[1] == random_subject, 0].values.tolist()
    random_relation = df_copy.loc[~df_copy[0].isin(related_relations), 0].sample(random_state=42).iloc[0]

    objects = []

    reference = find_reference(random_subject, random_relation, objects, df_copy)

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