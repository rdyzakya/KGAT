# entities.txt
# relations.txt
# texts.txt
# entities_alias.jsonl

# triples.json
# train.jsonl
# dev.jsonl
# test.jsonl

from argparse import ArgumentParser
import pandas as pd
import os
import json
from tqdm import tqdm

def init_args():
    parser = ArgumentParser()
    parser.add_argument("--dir", nargs='+', type=str)
    parser.add_argument("--out", type=str, required=True)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = init_args()

    os.makedirs(args.out, exist_ok=True)
    
    all_entities_txt = []
    all_relations_txt = []
    all_texts_txt = []
    all_entities_alias = []
    all_triples = []
    all_train = []
    all_dev = {}
    all_test = {}

    for i, el in tqdm(enumerate(args.dir)):
        # READ
        entities = open(os.path.join(el, "entities.txt"), 'r', encoding="utf-8").read().strip().splitlines()
        relations = open(os.path.join(el, "relations.txt"), 'r', encoding="utf-8").read().strip().splitlines()
        texts = open(os.path.join(el, "texts.txt"), 'r', encoding="utf-8").read().strip().splitlines()

        entities_alias = pd.read_json(os.path.join(el, "entities_alias.jsonl"), lines=True)

        triples = json.load(open(os.path.join(el, "triples.json"), 'r'))

        train = pd.read_json(os.path.join(el, "train.jsonl"), lines=True)
        dev = pd.read_json(os.path.join(el, "dev.jsonl"), lines=True)
        test = pd.read_json(os.path.join(el, "test.jsonl"), lines=True)

        # PREV DATA
        len_entities_before = len(all_entities_txt)
        len_relations_before = len(all_relations_txt)
        len_texts_before = len(all_texts_txt)
        len_triples_before = len(all_triples)

        # TXT
        all_entities_txt.extend(entities)
        all_relations_txt.extend(relations)
        all_texts_txt.extend(texts)

        # ENTITIES_ALIAS
        entities_alias.alias_idx = entities_alias.alias_idx.apply(lambda x: [el + len_entities_before for el in x])
        all_entities_alias.append(entities_alias)

        # TRIPLES
        all_triples.extend(triples)

        # TRAIN, DEV, SPLIT
        def incr_df(df):
            def handle_none(val, incr):
                if val is None:
                    return None
                return val + incr
            df.text = df.text.apply(lambda x: [el + len_texts_before for el in x])
            df.subject = df.subject.apply(lambda x: handle_none(x, len_entities_before))
            df.relation = df.relation.apply(lambda x: handle_none(x, len_relations_before))
            df.objects = df.objects.apply(lambda x: [el + len_entities_before for el in x])
            df.triple = df.triple.apply(lambda x: [el + len_triples_before for el in x])
            return df
        
        train = incr_df(train)
        dev = incr_df(dev)
        test = incr_df(test)

        all_train.append(train)
        all_dev[os.path.split(el)[-1]] = dev
        all_test[os.path.split(el)[-1]] = test
    
    all_entities_alias = pd.concat(all_entities_alias)
    all_train = pd.concat(all_train)

    # SAVE
    all_entities_txt = '\n'.join(all_entities_txt)
    with open(os.path.join(args.out, "entities.txt"), 'w', encoding="utf-8") as fp:
        fp.write(all_entities_txt)
    
    all_relations_txt = '\n'.join(all_relations_txt)
    with open(os.path.join(args.out, "relations.txt"), 'w', encoding="utf-8") as fp:
        fp.write(all_relations_txt)
    
    all_texts_txt = '\n'.join(all_texts_txt)
    with open(os.path.join(args.out, "texts.txt"), 'w', encoding="utf-8") as fp:
        fp.write(all_texts_txt)

    all_entities_alias.to_json(os.path.join(args.out, "entities_alias.jsonl"), orient="records", lines=True)

    with open(os.path.join(args.out, "triples.json"), 'w', encoding="utf-8") as fp:
        json.dump(all_triples, fp)

    all_train.to_json(os.path.join(args.out, "train.jsonl"), orient="records", lines=True)

    for k, v in all_dev.items():
        v.to_json(os.path.join(args.out, f"{k}.dev.jsonl"), orient="records", lines=True)
    for k, v in all_test.items():
        v.to_json(os.path.join(args.out, f"{k}.test.jsonl"), orient="records", lines=True)