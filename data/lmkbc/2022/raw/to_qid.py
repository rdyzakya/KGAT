import sys
sys.path.append("../../../../src")
import pandas as pd
from disambiguation import my_disambiguation
from tqdm import tqdm
import re


df = pd.read_json("./dev.jsonl", lines=True)

bar = tqdm(total=df.shape[0], desc="Processing")

def to_qid(row):
    object_entities = row["ObjectEntities"]
    qids = []
    for el1 in object_entities:
        current_qid = []
        for el2 in el1:
            qid = my_disambiguation(el2)
            if qid not in current_qid and re.match(r"Q\d+", qid):
                current_qid.append(qid)
        if len(current_qid) > 0:
            qids.append(current_qid)
    bar.update(1)
    return qids

df["ObjectEntitiesID"] = df.apply(to_qid, axis=1)
df.to_json("dev2.jsonl", orient="records", lines=True)