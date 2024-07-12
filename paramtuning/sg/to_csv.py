import json
import os
import pandas as pd
from tqdm import tqdm

def data_to_row(json_dict):
    result = {}
    if "hparam" in json_dict.keys():
        for k, v in json_dict["hparam"].items():
            result[k] = v
    for k, v in json_dict["evaluation_metrics"].items():
        result[k] = v
    for i, h in enumerate(json_dict["history"]):
        for k, v in h.items():
            if k == "epoch":
                continue
            result[f"epoch_{i+1}_{k}"] = v
    return result

def save_iter(folder):
    # iter 1
    df = []
    for fname in tqdm(os.listdir(folder)):
        path = os.path.join(folder, fname)
        json_dict = json.load(open(path))
        row = data_to_row(json_dict)
        df.append(row)

    df = pd.DataFrame(df)

    df.to_csv(f"./{folder}.csv", index=False)

    best = df.loc[df["test_sg_f1"].argmax()]

    with open(f"./best_{folder}.json", 'w') as fp:
        json.dump(best.to_dict(), fp)

save_iter("iter1")
save_iter("iter2")
save_iter("iter3")
save_iter("iter4")