import json
import pandas as pd
import os

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

def convert(ds, out_path):
    res = {}

    id2entity = list(set(ds[0].values.tolist() + ds[2].values.tolist()))
    id2entity = {i : [el] for i, el in enumerate(id2entity)}
    entity2id = {v[0] : k for k, v in id2entity.items()}
    
    id2relation = ds[1].unique().tolist()
    id2relation = {i : el for i, el in enumerate(id2relation)}
    relation2id = {v : k for k, v in id2relation.items()}
    
    res["num_triplets"] = ds.shape[0]
    res["num_entities"] = len(id2entity)
    res["num_relations"] = len(id2relation)

    res["entity"] = id2entity
    res["relation"] = id2relation

    ds_copy = ds.copy()
    ds_copy[0] = ds_copy[0].apply(entity2id.get)
    ds_copy[1] = ds_copy[1].apply(relation2id.get)
    ds_copy[2] = ds_copy[2].apply(entity2id.get)

    ds_copy = ds_copy[[0,2,1]] # subject - object - relation

    coo = [list(el.values()) for el in ds_copy.to_dict(orient="records")]

    res["coo"] = coo

    with open(out_path, 'w', encoding="utf-8") as fp:
        json.dump(res, fp)

    return res

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

if not os.path.exists("./proc"):
    os.makedirs("./proc")

train = convert(train, "./proc/train.json")
val = convert(val, "./proc/val.json")
test = convert(test, "./proc/test.json")