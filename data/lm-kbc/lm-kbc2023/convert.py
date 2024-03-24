import pandas as pd
import json
import random
import numpy as np
from ordered_set import OrderedSet
from tqdm import tqdm
import os

random.seed(42)
np.random.seed(42)

def find_reference(sub_id, rel_id, obj_ids, df):
    n_triples = random.randint(5,50)
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

train = pd.read_json(path_or_buf="./raw/train.jsonl", lines=True)
val = pd.read_json(path_or_buf="./raw/val.jsonl", lines=True)
test = pd.read_json(path_or_buf="./raw/test.jsonl", lines=True)

df = pd.concat([train, val, test]).reset_index(drop=True)

entity2id = dict()
relation2id = dict()
triples = []
num_entities = 0
num_relations = 0

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
        relation2id[r] = num_relations
        num_relations += 1
    
    for obj in o:
        triples.append((entity2id[s], relation2id[r], entity2id[obj]))

triples_df = pd.DataFrame(triples)

ds = []

# empty and not empty
for i, row in tqdm(df.iterrows()):
    sub_id = entity2id[row["SubjectEntity"]]
    rel_id = relation2id[row["Relation"]]
    obj_ids = [entity2id[el] for el in row["ObjectEntities"] if len(el) > 0]

    reference = find_reference(sub_id, rel_id, obj_ids, triples_df)

    ds.append({
        "subject" : sub_id,
        "relation" : rel_id,
        "objects" : obj_ids,
        "reference" : reference
    })

train, val, test = ds[:len(train)], ds[len(train):len(train)+len(val)], ds[len(train)+len(val):]

n_entity = max(entity2id.values()) + 1
entities = [[] for i in range(n_entity)]
for k, v in entity2id.items():
    entities[v].append(k)
entities = [max(el, key=len) for el in entities] # take the longest

if not os.path.exists("./proc"):
    os.makedirs("./proc")

with open("./proc/entities.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(entities))

with open("./proc/relations.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(relation2id.keys()))

with open("./proc/triples.json", 'w') as fp:
    json.dump(triples, fp)

with open("./proc/train.json", 'w') as fp:
    json.dump(train, fp)

with open("./proc/val.json", 'w') as fp:
    json.dump(val, fp)

with open("./proc/test.json", 'w') as fp:
    json.dump(test, fp)