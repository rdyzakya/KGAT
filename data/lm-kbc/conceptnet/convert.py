import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def convert(ds, out_path):
    res = {}

    id2entity = list(set(ds[1].values.tolist() + ds[2].values.tolist()))
    id2entity = {i : [el] for i, el in enumerate(id2entity)}
    entity2id = {v[0] : k for k, v in id2entity.items()}
    
    id2relation = ds[0].unique().tolist()
    id2relation = {i : el for i, el in enumerate(id2relation)}
    relation2id = {v : k for k, v in id2relation.items()}
    
    res["num_triplets"] = ds.shape[0]
    res["num_entities"] = len(id2entity)
    res["num_relations"] = len(id2relation)

    res["entity"] = id2entity
    res["relation"] = id2relation

    ds_copy = ds.copy()
    ds_copy[0] = ds_copy[0].apply(relation2id.get)
    ds_copy[1] = ds_copy[1].apply(entity2id.get)
    ds_copy[2] = ds_copy[2].apply(entity2id.get)

    ds_copy = ds_copy[[1,2,0]] # subject - object - relation

    coo = [list(el.values()) for el in ds_copy.to_dict(orient="records")]

    res["coo"] = coo

    with open(out_path, 'w', encoding="utf-8") as fp:
        json.dump(res, fp)

    return res

path = "./raw/data_preprocessed_release/cpnet/conceptnet.en.csv"

df = pd.read_csv(path, sep="\t", header=None)
df = df.fillna(value="nan")

train, val_test = train_test_split(df, test_size=0.2, random_state=42)
val, test = train_test_split(df, test_size=0.5, random_state=42)

if not os.path.exists("./proc"):
    os.makedirs("./proc")

train = convert(train, "./proc/train.json")
val = convert(val, "./proc/val.json")
test = convert(test, "./proc/test.json")