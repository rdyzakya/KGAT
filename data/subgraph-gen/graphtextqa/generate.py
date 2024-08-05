import pandas as pd
from tqdm import tqdm
import json
import requests
import re

unk = 0

def get_wikidata_id(query):
    search_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": query
    }
    
    response = requests.get(search_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if 'search' in data:
            if len(data['search']) > 0:
                return data['search'][0]['id']
    return None

def my_disambiguation(input_str):
    # Check if the string is an integer
    try:
        return int(input_str)
    except ValueError:
        pass

    # If not an integer, try to get the Wikidata ID
    wikidata_id = get_wikidata_id(input_str)
    if wikidata_id:
        return wikidata_id
    
    # If all else fails, return the original string
    return input_str

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

def process(df):
    global unk

    pair = []
    for i, row in tqdm(df.iterrows()):
        text = row["question"]

        entity_ids = row["subgraph"]["entities"]
        relation_ids = row["subgraph"]["relations"]

        triple = row["subgraph"]["adjacency"]
        entities = row["subgraph"]["entity_labels"]
        relations = row["subgraph"]["relation_labels"]

        pair_entry = {
            "text" : [],
            "subject" : None,
            "relation" : None,
            "objects" : row["answers"],
            "triple" : [],
        }

        is_null = False
        # entities
        for i, e in enumerate(entities):
            if e is None:
                while True:
                    try:
                        e = get_wikidata_entity_name(entity_ids[i])
                        break
                    except requests.exceptions.ConnectTimeout:
                        pass
            if e is None:
                is_null = True
            else:
                entities[i] = e
                if e not in all_entities:
                    all_entities[e] = len(all_entities)
        
        for eid, e in zip(entity_ids, entities):
            if e is None:
                continue
            if eid not in all_qids:
                all_qids[eid] = len(all_qids)
            if eid not in entities_alias:
                entities_alias[eid] = []
            if e not in entities_alias[eid]:
                entities_alias[eid].append(e)
            if e not in entity2qid:
                entity2qid[e] = eid
        
        # text
        if not is_null:
            if text not in all_texts:
                all_texts[text] = len(all_texts)
            if all_texts[text] not in pair_entry["text"]:
                pair_entry["text"].append(all_texts[text])
        
        # relations
        for r in relations:
            if r not in all_relations:
                all_relations[r] = len(all_relations)
        
        # triple
        for t in triple:
            if entities[t[0]] is None or entities[t[2]] is None:
                continue
            t = (
                all_qids[entity_ids[t[0]]], 
                all_relations[relations[t[1]]], 
                all_qids[entity_ids[t[2]]]
                )
            if t not in all_triples:
                all_triples[t] = len(all_triples)
            pair_entry["triple"].append(all_triples[t])
        
        if not is_null:
            pair.append(pair_entry)

    for i, row in tqdm(enumerate(pair)):
        answers = row["objects"]
        for j, a in enumerate(answers):
            if a not in all_entities:
                all_entities[a] = len(all_entities)
            qid = entity2qid.get(a, f"UNK_{unk}")
            if a not in entity2qid:
                entity2qid[a] = qid
                unk += 1
            if qid not in all_qids:
                all_qids[qid] = len(all_qids)
            if qid not in entities_alias:
                entities_alias[qid] = []
            if a not in entities_alias[qid]:
                entities_alias[qid].append(a)
            pair[i]["objects"][j] = all_qids[entity2qid[a]]
    return pair

train = pd.read_json("./raw/train.jsonl", lines=True)
dev = pd.read_json("./raw/dev.jsonl", lines=True)
test = pd.read_json("./raw/test.jsonl", lines=True)

all_texts = dict()

all_entities = dict()
all_qids = dict()
entity2qid = dict()
entities_alias = dict()

all_relations = dict()
all_triples = dict()

train_pair = process(train)
dev_pair = process(dev)
test_pair = process(test)

def dump(obj):
    result = [None for _ in range(len(obj))]

    for k, v in obj.items():
        result[v] = k
    
    return result

dump_texts = dump(all_texts)
dump_entities = dump(all_entities)
dump_relations = dump(all_relations)
dump_triples = dump(all_triples)

dump_texts = '\n'.join(dump_texts)
dump_entities = '\n'.join(dump_entities)
dump_relations = '\n'.join(dump_relations)

with open("texts.txt", 'w', encoding="utf-8") as fp:
    fp.write(dump_texts)
with open("entities.txt", 'w', encoding="utf-8") as fp:
    fp.write(dump_entities)
with open("relations.txt", 'w', encoding="utf-8") as fp:
    fp.write(dump_relations)

with open("triples.json", 'w') as fp:
    json.dump(dump_triples, fp)

pd.DataFrame(train_pair).to_json("train.jsonl", orient="records", lines=True)
pd.DataFrame(dev_pair).to_json("dev.jsonl", orient="records", lines=True)
pd.DataFrame(test_pair).to_json("test.jsonl", orient="records", lines=True)

alias = [None for i in range(len(all_qids))]
for k, v in all_qids.items():
    alias[v] = {"id" : k, "alias_idx" : [all_entities[el] for el in entities_alias[k]]}

pd.DataFrame(alias).to_json("entities_alias.jsonl", orient="records", lines=True)