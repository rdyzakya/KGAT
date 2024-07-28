import pandas as pd
import requests
from tqdm import tqdm
import json

train = pd.read_json("./raw/train.jsonl", lines=True)
dev = pd.read_json("./raw/val.jsonl", lines=True)
test = pd.read_json("./raw/test.jsonl", lines=True)

split_type = [0 for _ in range(len(train))] + [1 for _ in range(len(dev))] + [2 for _ in range(len(test))]

all_df = pd.concat([train, dev, test])
all_df = all_df.reset_index(drop=True)
all_df["split"] = split_type

def get_wikidata_entity_name(entity_id):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "format": "json",
        "languages": "en"
    }
    
    response = requests.get(url, params=params)
    data = response.json()

    if 'entities' in data and entity_id in data['entities']:
        entity = data['entities'][entity_id]
        if 'labels' in entity and 'en' in entity['labels']:
            return entity['labels']['en']['value']
    return None

alias = {} # qid to entity
modified_obj_entities = []
for i, row in tqdm(all_df.iterrows()):
    if row["SubjectEntityID"] not in alias:
        alias[row["SubjectEntityID"]] = []
        
    if row["SubjectEntity"] not in alias[row["SubjectEntityID"]]:
        alias[row["SubjectEntityID"]].append(row["SubjectEntity"])
    
    object_entities = row["ObjectEntities"]
    if len(row["ObjectEntitiesID"]) != len(row["ObjectEntities"]):
        object_entities = [get_wikidata_entity_name(el) for el in row["ObjectEntitiesID"]]
        all_df.loc[i, "ObjectEntities"] = str(object_entities)
        modified_obj_entities.append(i)
    for el1, el2 in zip(row["ObjectEntitiesID"],object_entities):
        if len(el1) == 0:
            continue
        if el1 not in alias:
            alias[el1] = []
        
        if el2 not in alias[el1]:
            alias[el1].append(el2)

all_df.loc[modified_obj_entities, "ObjectEntities"] = all_df.loc[modified_obj_entities, "ObjectEntities"].apply(eval)

entities = []
for k, v in alias.items():
    for el in v:
        entities.append(el)

relations = all_df.Relation.unique().tolist()

entitie_map = {el : i for i, el in enumerate(entities)}
relations_map = {el : i for i, el in enumerate(relations)}

alias_jsonl = [{"id" : k, "alias_idx" : [entitie_map[el] for el in v]} for k, v in alias.items()]

qid_map = {}
for i, el in enumerate(alias_jsonl):
    qid_map[el["id"]] = i

entities_to_qid_map = {}
for k, v in alias.items():
    for el in v:
        entities_to_qid_map[el] = qid_map[k]

pd.DataFrame(alias_jsonl).to_json("entities_alias.jsonl", orient="records", lines=True)

all_texts = {}
all_triples = {}

train_dump = []
dev_dump = []
test_dump = []

for i, row in all_df.iterrows():
    entry = {}
    text = f'subject : {row["SubjectEntity"]} | relation : {row["Relation"]}'
    if text not in all_texts:
        all_texts[text] = len(all_texts)
    entry["text"] = all_texts[text]

    s = entities_to_qid_map[row["SubjectEntity"]]
    r = relations_map[row["Relation"]]

    entry["subject"] = s
    entry["relation"] = r
    entry["objects"] = []
    entry["triple"] = []

    for o2 in row["ObjectEntities"]:
        if len(o2) == 0:
            continue
        o = entities_to_qid_map[o2]
        triple = (s, r, o)
        if triple not in all_triples:
            all_triples[triple] = len(all_triples)
        if all_triples[triple] not in entry["triple"]:
            entry["triple"].append(all_triples[triple])
        if o not in entry["objects"]:
            entry["objects"].append(o)
    if row["split"] == 0:
        train_dump.append(entry)
    elif row["split"] == 1:
        dev_dump.append(entry)
    elif row["split"] == 2:
        test_dump.append(entry)

with open("entities.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(entities))
with open("relations.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(relations))

pd.DataFrame(train_dump).to_json("train.jsonl", index=False, orient="records", lines=True)
pd.DataFrame(dev_dump).to_json("dev.jsonl", index=False, orient="records", lines=True)
pd.DataFrame(test_dump).to_json("test.jsonl", index=False, orient="records", lines=True)

reverse_all_triples = {v : k for k, v in all_triples.items()}
all_triples = [reverse_all_triples[i] for i in range(len(all_triples))]

reverse_all_texts = {v : k for k, v in all_texts.items()}
all_texts = [reverse_all_texts[i] for i in range(len(all_texts))]

with open("./triples.json", 'w') as fp:
    json.dump(all_triples, fp)

with open("./texts.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(all_texts))