import random
import os
import json

entity_map_path = "./raw/entity2text.txt"
relation_map_path = "./raw/relation2text.txt"
entity_map_path2 = "./raw/entity2textlong.txt"
relation_map_path2 = "./raw/relation2textlong.txt"
triplets_path = "./raw/wiki_tuple_ids.txt"

def load_data(path):
    with open(path, 'r', encoding="utf-8") as fp:
        data = fp.read().strip().splitlines()
    return data

entity_map = load_data(entity_map_path)
entity_map = [el.split('\t') for el in entity_map]
entity_map = {el[0] : el[1] for el in entity_map}

entity_map2 = load_data(entity_map_path2)
entity_map2 = [el.split('\t') for el in entity_map2]
entity_map2 = {el[0] : el[1] for el in entity_map2}

relation_map = load_data(relation_map_path)
relation_map = [el.split('\t') for el in relation_map]
relation_map = {el[0] : el[1] for el in relation_map}

relation_map2 = load_data(relation_map_path2)
relation_map2 = [el.split('\t') for el in relation_map2]
relation_map2 = {el[0] : el[1] for el in relation_map2}

entity_map = {
    k : list(set([entity_map[k], entity_map2[k]])) for k in entity_map.keys()
}

relation_map = {
    k : list(set([relation_map[k], relation_map2[k]])) for k in relation_map.keys()
}

triplets = load_data(triplets_path)
random.seed(42)
random.shuffle(triplets)
triplets = [el.split('\t') for el in triplets]

train_split = int(0.6 * len(triplets))
val_split = train_split + int(0.2 * len(triplets))
train, val, test = triplets[:train_split], triplets[train_split:val_split], triplets[val_split:]

def create_ds(entity_map, relation_map, triplets):
    unique_entities = set()
    unique_relations = set()
    for h, r, t in triplets:
        unique_entities.add(h)
        unique_entities.add(t)
        unique_relations.add(r)
    
    num_entities = len(unique_entities)
    num_relations = len(unique_relations)

    unique_entities = {k : i for i, k in enumerate(unique_entities)}
    unique_relations = {k : i for i, k in enumerate(unique_relations)}

    entity_id2text = {i : entity_map[k] for k, i in unique_entities.items()}
    relation_id2text = {i : relation_map[k] for k, i in unique_relations.items()}

    coo = [
        [unique_entities[el[0]], unique_entities[el[2]], unique_relations[el[1]]] for el in triplets
    ]

    return {
        "num_triplets" : len(coo),
        "num_entities" : num_entities,
        "num_relations" : num_relations,
        "entity" : entity_id2text,
        "relation" : relation_id2text,
        "coo" : coo
    }

train_ds = create_ds(entity_map, relation_map, train)
val_ds = create_ds(entity_map, relation_map, val)
test_ds = create_ds(entity_map, relation_map, test)

if not os.path.exists("./proc"):
    os.makedirs("./proc")

with open("./proc/train.json", 'w') as fp:
    json.dump(train_ds, fp)

with open("./proc/val.json", 'w') as fp:
    json.dump(val_ds, fp)

with open("./proc/test.json", 'w') as fp:
    json.dump(test_ds, fp)