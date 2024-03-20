import os
import json
import pandas as pd
import random
from tqdm import tqdm

random.seed(42)

def find_reference(sub_id, rel_id, df):
    n_triples = random.randint(20,500)
    res = set()
    while len(res) < n_triples:
        decider = random.random()
        if decider <= 0.25: # same entity
            same_sub = df.loc[(df[1] == sub_id) | (df[2] == sub_id)]
            res = res.union(same_sub.index.to_list())
        elif decider <= 0.5: # same relation
            same_rel = df.loc[df[0] == rel_id]
            res = res.union(same_rel.index.to_list())
        elif decider <= 0.75: # tailing
            if len(res) == 0:
                continue
            res_triples = df.loc[list(res)]

            res_nodes = res_triples[1].values.tolist() + res_triples[2].values.tolist()
            if sub_id in res_nodes:
                res_nodes.remove(sub_id)

            k = random.randint(1,len(res_nodes))
            chosen_nodes = random.choices(res_nodes, k=k)

            related_triples = df.loc[df[1].isin(chosen_nodes) | df[2].isin(chosen_nodes)]
            res = res.union(related_triples.index.to_list())
        else: #random
            n_random_triples = random.randint(1, n_triples - len(res))
            samples = df.sample(n_random_triples, random_state=42)
            res = res.union(samples.index.to_list())
    return res

path = "./raw/data_preprocessed_release/cpnet/conceptnet.en.csv"

df = pd.read_csv(path, sep="\t", header=None)
df = df.fillna(value="nan")

entities = list(set(df[1].values.tolist() + df[2].values.tolist()))
entity2id = {el : i for i, el in enumerate(entities)}

relations = df[0].unique().tolist()
relation2id = {el : i for i, el in enumerate(relations)}

df_copy = df.copy()
df_copy[0] = df_copy[0].apply(relation2id.get)
df_copy[1] = df_copy[1].apply(entity2id.get)
df_copy[2] = df_copy[2].apply(entity2id.get)

df_copy = df_copy[[1,2,0]] # subject - object - relation

triples = [list(el.values()) for el in df_copy.to_dict(orient="records")]

df_copy["done"] = False
ds = []

# not empty
print(f"Processing {df_copy.shape[0]} iterations...")
for i, row in tqdm(df_copy.iterrows()):
    if row["done"]:
        continue
    sub_rel = df_copy.loc[(df_copy[0] == row[0]) & (df_copy[1] == row[1])]
    df_copy.loc[sub_rel.index, "done"] = True
    
    objects = sub_rel[2].values.tolist()

    reference = find_reference(row[1], row[0], df_copy)

    ds.append({
        "subject" : row[1],
        "relation" : row[0],
        "objects" : objects,
        "reference" : reference
    })

# empty, 15% from the not empty
print(f"Processing {int(0.15 * len(ds))} iterations...")
for i in tqdm(range(int(0.15 * len(ds)))):
    random_subject = random.choice([df[1].sample().iloc[0], df[2].sample().iloc[0]])
    related_relations = df.loc[df[1] == random_subject, 0].values.tolist()
    random_relation = df.loc[~df[0].isin(related_relations), 0].sample().iloc[0]

    reference = find_reference(row[1], row[0], df_copy)

    ds.append({
        "subject" : random_subject,
        "relation" : random_relation,
        "objects" : [],
        "reference" : reference
    })

random.shuffle(ds)

train, val, test = ds[:int(len(ds) * 0.8)], ds[int(len(ds) * 0.8), int(len(ds) * 0.9)], ds[int(len(ds) * 0.9):]

if not os.path.exists("./proc"):
    os.makedirs("./proc")

with open("./proc/entities.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(entities))

with open("./proc/relations.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(relations))

with open("./proc/triples.json", 'w') as fp:
    json.dump(triples, fp)

with open("./proc/train.json", 'w') as fp:
    json.dump(train, fp)

with open("./proc/val.json", 'w') as fp:
    json.dump(val, fp)

with open("./proc/test.json", 'w') as fp:
    json.dump(test, fp)