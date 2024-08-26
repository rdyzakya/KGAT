import sys
sys.path.append("./src")
import json
import pandas as pd
import re
from utils import TRUE_FLAG, FALSE_FLAG, EMPTY_OBJECT
from disambiguation import my_disambiguation, get_wikidata_entity_name
from tqdm import tqdm
from io import StringIO

jsonl_path = "./data/lmkbc/2023/raw/test.jsonl"
preds_path = "./lmkbc2023-llama3.1/preds-test.json"
disambiguate = True

pattern = re.compile(rf"([^|]+)\s?\|\s?valid\s?:\s?({TRUE_FLAG}|{FALSE_FLAG})")

df = pd.read_json(jsonl_path, lines=True)
preds = json.load(open(preds_path, 'r'))


assert len(df) == len(preds)

res = []

for i in tqdm(range(len(df))):
    subject_entity = df.loc[i, "SubjectEntity"]
    relation = df.loc[i, "Relation"]
    
    row_preds = preds[i]

    text = row_preds["text"]
    score = row_preds["score"]

    predictions = {}
    for j in range(len(text)):
        m = pattern.match(text[j])
        if m is None:
            continue
        
        obj = m.group(1).strip()
        flag = m.group(2)

        if obj.lower() == EMPTY_OBJECT.lower():
            continue

        q_id = my_disambiguation(obj)

        if not re.match(r"Q\d+", str(q_id)) and not isinstance(q_id, int):
            continue

        scr = score[j]

        if q_id not in predictions:
            predictions[q_id] = {}
        if flag not in predictions[q_id]:
            predictions[q_id][flag] = scr
        else:
            predictions[q_id][flag] = max(scr, predictions[q_id][flag])
    
    obj_key = "ObjectEntitiesID" if disambiguate else "ObjectEntities"
    entry = {
        "SubjectEntity" : subject_entity,
        "Relation" : relation,
        obj_key : []
    }

    for k, v in predictions.items():
        flag = max(v, key=v.get)
        if flag == TRUE_FLAG:
            if not disambiguate and isinstance(k, str):
                entity_name = get_wikidata_entity_name(k)
                
                if entity_name:
                    entry[obj_key].append(entity_name.lower())
            else:
                entry[obj_key].append(str(k))
    
    res.append(entry)

with open("predictions-test-2023.jsonl", "w") as f:
    for row in res:
        f.write(json.dumps(row) + "\n")