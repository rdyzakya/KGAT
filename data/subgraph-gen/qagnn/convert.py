import os
import shutil
import pandas as pd
import pickle
import json
import numpy as np
from tqdm import tqdm

def open_pickle(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data

def dump_json(path, obj):
    with open(path, 'w', encoding="utf-8") as fp:
        json.dump(obj, fp)

# Make output directory
print("Create output directory...")
csqa_dir = "./proc/csqa"
obqa_dir = "./proc/obqa"

if not os.path.exists(csqa_dir):
    os.makedirs(csqa_dir)

if not os.path.exists(obqa_dir):
    os.makedirs(obqa_dir)

# Copy entity file
print("Copying entity file...")
shutil.copy("./raw/data_preprocessed_release/cpnet/concept.txt", os.path.join(csqa_dir, "entity.txt"))
shutil.copy("./raw/data_preprocessed_release/cpnet/concept.txt", os.path.join(obqa_dir, "entity.txt"))

with open(os.path.join(csqa_dir, "entity.txt"), 'r', encoding="utf-8") as fp:
    concepts = fp.read().strip().splitlines()

# Create relation file
print("Creating relation file...")
triples = pd.read_csv("./raw/data_preprocessed_release/cpnet/conceptnet.en.csv", sep='\t', header=None)

relations = triples[0].unique()

with open(os.path.join(csqa_dir, "relation.txt"), 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(sorted(relations)))
with open(os.path.join(obqa_dir, "relation.txt"), 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(sorted(relations)))

# Get statements
print("Get statements...")
## CSQA
train_statement_csqa = pd.read_json("./raw/data_preprocessed_release/csqa/statement/train.statement.jsonl", lines=True)
dev_statement_csqa = pd.read_json("./raw/data_preprocessed_release/csqa/statement/dev.statement.jsonl", lines=True)
test_statement_csqa = pd.read_json("./raw/data_preprocessed_release/csqa/statement/test.statement.jsonl", lines=True)

## OBQA
train_statement_obqa = pd.read_json("./raw/data_preprocessed_release/obqa/statement/train.statement.jsonl", lines=True)
dev_statement_obqa = pd.read_json("./raw/data_preprocessed_release/obqa/statement/dev.statement.jsonl", lines=True)
test_statement_obqa = pd.read_json("./raw/data_preprocessed_release/obqa/statement/test.statement.jsonl", lines=True)

# Get grounded
print("Get grounded...")
## CSQA
train_grounded_csqa = pd.read_json("./raw/data_preprocessed_release/csqa/grounded/train.grounded.jsonl", lines=True)
dev_grounded_csqa = pd.read_json("./raw/data_preprocessed_release/csqa/grounded/dev.grounded.jsonl", lines=True)
test_grounded_csqa = pd.read_json("./raw/data_preprocessed_release/csqa/grounded/test.grounded.jsonl", lines=True)

## OBQA
train_grounded_obqa = pd.read_json("./raw/data_preprocessed_release/obqa/grounded/train.grounded.jsonl", lines=True)
dev_grounded_obqa = pd.read_json("./raw/data_preprocessed_release/obqa/grounded/dev.grounded.jsonl", lines=True)
test_grounded_obqa = pd.read_json("./raw/data_preprocessed_release/obqa/grounded/test.grounded.jsonl", lines=True)

# Get graphs
print("Get graphs...")
## CSQA
train_graph_csqa = open_pickle("./raw/data_preprocessed_release/csqa/graph/train.graph.adj.pk")
dev_graph_csqa = open_pickle("./raw/data_preprocessed_release/csqa/graph/dev.graph.adj.pk")
test_graph_csqa = open_pickle("./raw/data_preprocessed_release/csqa/graph/test.graph.adj.pk")

## OBQA
train_graph_obqa = open_pickle("./raw/data_preprocessed_release/obqa/graph/train.graph.adj.pk")
dev_graph_obqa = open_pickle("./raw/data_preprocessed_release/obqa/graph/dev.graph.adj.pk")
test_graph_obqa = open_pickle("./raw/data_preprocessed_release/obqa/graph/test.graph.adj.pk")

# Create dataset
relation_indexs = [i for i in range(len(relations))] # sorted already
def create_ds(all_statement, all_grounded, all_graph):
    ds = []
    for _, row in tqdm(all_statement.iterrows(), desc="Processing dataset"):
        for statement in row["statements"]:
            if not statement["label"]:
                continue
            answer_statement = statement["statement"]
            graph_index = all_grounded.loc[all_grounded["sent"] == answer_statement].index[0]
            graph = all_graph[graph_index]

            concept_indexs = graph["concepts"]
            concept_indexs.sort()
            adjacency_matrix = graph["adj"].toarray()
            adjacency_matrix = adjacency_matrix.reshape(
                len(relation_indexs),
                len(concept_indexs),
                len(concept_indexs)
            )
            x_coo = np.argwhere(adjacency_matrix != 0)
            amask = graph["amask"]
            qmask = graph["qmask"]
            y_node_cls = concept_indexs[amask]
            y_node_cls = np.isin(concept_indexs, y_node_cls).astype(np.int32)
            y_coo = x_coo[amask[x_coo[:,1]] | amask[x_coo[:,2]]]
            
            y_coo[:,1] = concept_indexs[y_coo[:,1]]
            y_coo[:,2] = concept_indexs[y_coo[:,2]]

            x_coo[:,1] = concept_indexs[x_coo[:,1]]
            x_coo[:,2] = concept_indexs[x_coo[:,2]]

            concept_index_vectorize = np.vectorize(lambda x: np.where(concept_indexs == x)[0][0])

            if len(x_coo) > 0:
                x_coo[:,1] = concept_index_vectorize(x_coo[:,1])
                x_coo[:,2] = concept_index_vectorize(x_coo[:,2])
                x_coo = x_coo.T
                x_coo = x_coo[[1,0,2]] # subject - relation - object
                x_coo = x_coo.T
            
            y_coo_cls = [0 for el in x_coo]
            if len(y_coo) > 0:
                y_coo[:,1] = concept_index_vectorize(y_coo[:,1])
                y_coo[:,2] = concept_index_vectorize(y_coo[:,2])
                y_coo = y_coo.T
                y_coo = y_coo[[1,0,2]]
                y_coo = y_coo.T
                y_coo_cls = [int(el in y_coo) for el in x_coo]

            entry = {
                "text" : answer_statement,
                "entities" : concept_indexs.tolist(),
                "relations" : relation_indexs,
                "x_coo" : x_coo.tolist(),
                "y_coo_cls" : y_coo_cls,
                "y_node_cls" : y_node_cls.tolist()
            }
            ds.append(entry)

    return ds

## CSQA
print("Create CSQA dataset...")
train_ds_csqa = create_ds(train_statement_csqa, train_grounded_csqa, train_graph_csqa)
dump_json(os.path.join(csqa_dir, "train.json"), train_ds_csqa)
dev_ds_csqa = create_ds(dev_statement_csqa, dev_grounded_csqa, dev_graph_csqa)
dump_json(os.path.join(csqa_dir, "dev.json"), dev_ds_csqa)

## OBQA
print("Create OBQA dataset...")
train_ds_obqa = create_ds(train_statement_obqa, train_grounded_obqa, train_graph_obqa)
dump_json(os.path.join(obqa_dir, "train.json"), train_ds_obqa)
dev_ds_obqa = create_ds(dev_statement_obqa, dev_grounded_obqa, dev_graph_obqa)
dump_json(os.path.join(obqa_dir, "dev.json"), dev_ds_obqa)

print("Done!")