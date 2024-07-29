import pandas as pd
import requests
from tqdm import tqdm
import re
from ordered_set import OrderedSet
import json


# Disambiguation baseline
def disambiguation_baseline(item):
    try:
        # If item can be converted to an integer, return it directly
        return int(item)
    except ValueError:
        # If not, proceed with the Wikidata search
        try:
            url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
            data = requests.get(url).json()
            # Return the first id (Could upgrade this in the future)
            return data['search'][0]['id']
        except:
            return item

train = pd.read_json("./raw/train.jsonl", lines=True)
dev = pd.read_json("./raw/dev.jsonl", lines=True)
test = pd.read_json("./raw/test.jsonl", lines=True)

split_type = [0 for _ in range(len(train))] + [1 for _ in range(len(dev))] + [2 for _ in range(len(test))]

all_df = pd.concat([train, dev, test])
all_df = all_df.reset_index(drop=True)
all_df["split"] = split_type

alias = {}

# FROM SUBJECT
for i, row in all_df.iterrows():
    wikidata_id = f'Q{row["WikidataID"]}'
    if wikidata_id not in alias:
        alias[wikidata_id] = []
    alias[wikidata_id].append(row["SubjectEntity"])
# FROM OBJECTS
unk_entities = []

for i, row in tqdm(all_df.iterrows()):
    objects = row["ObjectEntities"]
    for el in objects:
        entry_unk_entities = []
        all_qid = []
        for a in el:
            qid = disambiguation_baseline(a)
            if qid in alias:
                # if not isin_lower(a, alias[qid]):
                if a not in alias[qid]:
                    alias[qid].append(a)
                all_qid.append(qid)
            elif re.match(r"Q\d+", qid):
                alias[qid] = [a]
                all_qid.append(qid)
            else:
                if len(all_qid) == 0:
                    entry_unk_entities.append(a)
                else:
                    alias[all_qid[0]].append(a)
        unk_entities.extend(entry_unk_entities)

for k, v in alias.items():
    alias[k] = list(OrderedSet(v))


unk_entities = list(OrderedSet(unk_entities))
for i, el in enumerate(unk_entities):
    qid = None
    for k, v in alias.items():
        if el in v:
            qid = k
            break
    if qid is None:
        qid = disambiguation_baseline(el)
    if not re.match(r"Q\d+", qid):
        alias[f"UNK{i}"] = [el]

entities = []
for k, v in alias.items():
    for el in v:
        entities.append(el)

relations = all_df.Relation.unique().tolist()

with open("./entities.txt", 'w', encoding="utf-8")as fp:
    fp.write('\n'.join(entities))

with open("./relations.txt", 'w', encoding="utf-8")as fp:
    fp.write('\n'.join(relations))

entities_map = {
    el : i for i, el in enumerate(entities)
}

relations_map = {
    el : i for i, el in enumerate(relations)
}

alias_jsonl = [{"id" : k, "alias_idx" : [entities_map[el] for el in v]} for k, v in alias.items()]

qid_map = {}
for i, el in enumerate(alias_jsonl):
    qid_map[el["id"]] = i

entities_to_qid_map = {}
for k, v in alias.items():
    for el in v:
        entities_to_qid_map[el] = qid_map[k]

pd.DataFrame(alias_jsonl).to_json("entities_alias.jsonl", orient="records", lines=True)

all_triples = {}
all_texts = {}

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
    
    for o1 in row["ObjectEntities"]:
        for o2 in o1:
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

reverse_all_triples = {v : k for k, v in all_triples.items()}
all_triples = [reverse_all_triples[i] for i in range(len(all_triples))]

reverse_all_texts = {v : k for k, v in all_texts.items()}
all_texts = [reverse_all_texts[i] for i in range(len(all_texts))]

pd.DataFrame(train_dump).to_json("train.jsonl", index=False, orient="records", lines=True)
pd.DataFrame(dev_dump).to_json("dev.jsonl", index=False, orient="records", lines=True)
pd.DataFrame(test_dump).to_json("test.jsonl", index=False, orient="records", lines=True)

with open("./triples.json", 'w') as fp:
    json.dump(all_triples, fp)

with open("./texts.txt", 'w', encoding="utf-8") as fp:
    fp.write('\n'.join(all_texts))