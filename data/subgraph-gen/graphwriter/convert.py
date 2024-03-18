import pandas as pd

def preprocess(df, rel_index):
    result = []
    for i, row in df.iterrows():
        entities = row[1].split(" ; ")
        masks = row[2].split()
        assert len(entities) == len(masks)

        triplets = row[3].split(" ; ")
        y_coo = []
        for t in triplets:
            s, r, o = t.split()
            s, r, o = int(s), int(r), int(o)
            y_coo.append([s,o,r]) # subject - object - relation
        
        masks = [el[:-1] + f"_{i}>" for i, el in enumerate(masks)]
        text = row[4]
        for i, m in enumerate(masks):
            text.replace(m, entities[i])
        result.append({
            "text" : text,
            "entities" : entities,
            "relations" : rel_index
        })

train_path = "./raw/preprocessed.train.tsv"
val_path = "./raw/preprocessed.val.tsv"
test_path = "./raw/preprocessed.test.tsv"

rel_path = "./raw/relations.vocab"

train = pd.read_csv(train_path, sep='\t', header=None)
val = pd.read_csv(val_path, sep='\t', header=None)
test = pd.read_csv(test_path, sep='\t', header=None)

with open(rel_path, 'r') as fp:
    rel = fp.read().strip().splitlines()
