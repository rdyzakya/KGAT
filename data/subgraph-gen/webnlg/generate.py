import pandas as pd
from tqdm import tqdm
import json

train = pd.read_json("./raw/train.jsonl", lines=True)
dev = pd.read_json("./raw/dev.jsonl", lines=True)
test = pd.read_json("./raw/test.jsonl", lines=True)

all_texts = dict()
all_entities = dict()
all_relations = dict()
all_triples = dict()

def process(df):
    pair = []
    for i, row in tqdm(df.iterrows()):
        modified_triple_sets = row["modified_triple_sets"]["mtriple_set"][0]
        text = row["lex"]["text"]

        pair_entry = {
            "text" : [],
            "subject" : None,
            "relation" : None,
            "objects" : [],
            "triple" : [],
        }
        
        for t in text:
            if t not in all_texts:
                all_texts[t] = len(all_texts)
            if all_texts[t] not in pair_entry["text"]:
                pair_entry["text"].append(all_texts[t])

        for t in modified_triple_sets:
            s, r, o = t.split('|')
            s = s.strip()
            r = r.strip()
            o = o.strip()

            if s not in all_entities:
                all_entities[s] = len(all_entities)
            if r not in all_relations:
                all_relations[r] = len(all_relations)
            if o not in all_entities:
                all_entities[o] = len(all_entities)
            
            triple = (all_entities[s], all_relations[r], all_entities[o])

            if triple not in all_triples:
                all_triples[triple] = len(all_triples)
            
            if all_triples[triple] not in pair_entry["triple"]:
                pair_entry["triple"].append(all_triples[triple])
        
        pair.append(pair_entry)
    return pair

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

with open("texts.txt", 'w') as fp:
    fp.write(dump_texts)
with open("entities.txt", 'w') as fp:
    fp.write(dump_entities)
with open("relations.txt", 'w') as fp:
    fp.write(dump_relations)

with open("triples.json", 'w') as fp:
    json.dump(dump_triples, fp)

pd.DataFrame(train_pair).to_json("train.jsonl", orient="records", lines=True)
pd.DataFrame(dev_pair).to_json("dev.jsonl", orient="records", lines=True)
pd.DataFrame(test_pair).to_json("test.jsonl", orient="records", lines=True)

# create alias mapping
# since it is not provided by webnlg then,

alias = [{"id" : i , "alias_idx" : [i]} for i in range(len(all_entities))]
pd.DataFrame(alias).to_json("entities_alias.jsonl", orient="records", lines=True)