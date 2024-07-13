from torch.utils.data import Dataset
from ..utils import Mask, NULL_SYM
import torch
import json
import re
import warnings
import random
from tqdm import tqdm

# def apply_template(text, subject, relation, objects=None):
#     text = text.replace(Mask.SUBJECT_MASK, subject).replace(Mask.RELATION_MASK, relation)
#     if objects is None:
#         return text
#     return text.replace(Mask.OBJECT_MASK, str(objects))

def load_json(path):
    with open(path, 'r', encoding="utf-8") as fp:
        data = json.load(fp)
    return data

def load_id2map(path):
    with open(path, 'r', encoding="utf-8") as fp:
        data = fp.read().strip().splitlines()
    data = {i : el for i, el in enumerate(data)}
    return data

class SubgraphGenerationDataset(Dataset):
    def __init__(self, path, id2entity, id2relation, start_index=0, n_data=None, split_size=300):
        # random.seed(random_state)
        data = load_json(path)
        new_data = []
        print("Before split by coo : ", len(data))
        for entry in tqdm(data):
            new_data.extend(
                self.split_by_coo(entry, split_size=split_size)
            )
        print("After split by coo : ", len(new_data))
        random.shuffle(new_data)
        end_index = start_index + n_data if n_data else -1
        self.data = new_data[start_index:end_index]
        self.id2entity = id2entity
        self.id2relation = id2relation
        # random.seed(None)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        # text, entities, relations, x_coo, y_coo/y_node_cls
        text = self.data[i]["text"]

        entities = self.data[i]["entities"]
        entities = [self.id2entity[e] for e in entities]

        relations = self.data[i]["relations"]
        relations = [self.id2relation[r] for r in relations]

        x_coo = torch.tensor(self.data[i]["x_coo"]) # 0 - subject ; 1 - relation ; 2 - object

        y_coo_cls = torch.tensor(self.data[i]["y_coo_cls"])
        # y_coo_mask = [bool(el) for el in y_coo_cls]
        # y_coo = x_coo[y_coo_mask]

        x_coo = x_coo.T
        # y_coo = y_coo.T

        return text, entities, relations, x_coo, y_coo_cls
    
    def split_by_coo(self, entry, split_size=300):
        # text = entry["text"]
        # entities = entry["entities"]
        # relations = entry["relations"]
        # x_coo = entry["x_coo"]
        # y_coo_cls = entry["y_coo_cls"]
        # y_node_cls = entry["y_node_cls"]

        assert len(entry["x_coo"]) == len(entry["y_coo_cls"])
        assert len(entry["entities"]) == len(entry["y_node_cls"])

        if len(entry["x_coo"]) <= split_size:
            return [entry]
        # else
        coo_index = [i for i in range(len(entry["x_coo"]))]
        random.shuffle(coo_index)
        chunks = [coo_index[i:i+split_size] for i in range(0,len(entry["x_coo"]),split_size)]

        return [self.create_new_entry_from_chunk(c, entry) for c in chunks]
    
    def create_new_entry_from_chunk(self, chunk, entry):
        text = entry["text"]
        entities = entry["entities"]
        relations = entry["relations"]
        x_coo = entry["x_coo"]
        y_coo_cls = entry["y_coo_cls"]
        y_node_cls = entry["y_node_cls"]


        result_x_coo = [x_coo[i] for i in chunk]
        result_y_coo_cls = [y_coo_cls[i] for i in chunk]

        # entity_idx = 0
        entity_map = {}
        relation_map = {}

        for i in range(len(result_x_coo)):
            entity1 = result_x_coo[i][0]
            entity2 = result_x_coo[i][2]

            if entity1 not in entity_map.keys():
                entity_map[entity1] = len(entity_map)
            if entity2 not in entity_map.keys():
                entity_map[entity2] = len(entity_map)

            relation = result_x_coo[i][1]
            if relation not in relation_map.keys():
                relation_map[relation] = len(relation_map)
            
            result_x_coo[i] = [
                entity_map[entity1],
                relation_map[relation],
                entity_map[entity2]
            ]
        
        inverse_entity_map = {v : k for k, v in entity_map.items()}
        inverse_relation_map = {v : k for k, v in relation_map.items()}

        result_entities = [entities[inverse_entity_map[i]] for i in range(len(inverse_entity_map))]
        result_relations = [relations[inverse_relation_map[i]] for i in range(len(inverse_relation_map))]
        result_y_node_cls = [y_node_cls[inverse_entity_map[i]] for i in range(len(inverse_entity_map))]

        return dict(
            text = text,
            entities = result_entities,
            relations = result_relations,
            x_coo = result_x_coo,
            y_coo_cls = result_y_coo_cls,
            y_node_cls = result_y_node_cls
        )
            

class LMKBCDataset(Dataset):
    def __init__(self, 
                 path, 
                 id2entity, 
                 id2relation, 
                 triples, 
                 prompt_template=f"{Mask.KG_MASK} -> S : {Mask.SUBJECT_MASK} | R : {Mask.RELATION_MASK} | O : {Mask.OBJECT_MASK}", 
                 graph_query_template=f"S : {Mask.SUBJECT_MASK} | R : {Mask.RELATION_MASK}", 
                 n_virtual_token=1,
                 test=False,
                 n_data=None,
                 start_index=0):
        self.data = load_json(path)
        end_index = start_index + n_data if n_data else -1
        self.data = self.data[start_index:end_index]
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.triples = triples
        self.prompt_template = prompt_template
        self.graph_query_template = graph_query_template
        self.n_virtual_token = n_virtual_token
        self.test = test

        if len(re.findall(Mask.KG_MASK, prompt_template)) != n_virtual_token:
            new_prompt_template = prompt_template.replace(Mask.KG_MASK, ''.join([Mask.KG_MASK for _ in range(n_virtual_token)]))
            warnings.warn(f"Number of knowledge graph mask in your template is not the same with `n_virtual_token`, we transform it from {prompt_template} to {new_prompt_template}")
            assert len(re.findall(Mask.KG_MASK, new_prompt_template)) == n_virtual_token, f"Please make the knowledge graph mask in a contiguous manner like `{Mask.KG_MASK}{Mask.KG_MASK} <- 2 masks`"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        # text_in, entities, relations, x_coo, text_out
        subject_id = self.data[i]["subject"]
        relation_id = self.data[i]["relation"]
        object_ids = self.data[i]["objects"]
        reference = self.data[i]["reference"]

        subject = self.id2entity[subject_id]
        relation = self.id2relation[relation_id]
        objects = [self.id2entity[el] for el in object_ids] # TODO if empty?
        objects = [NULL_SYM] if len(objects) == 0 else objects # empty object

        sr_graph_query = (self.graph_query_template
                          .replace(Mask.SUBJECT_MASK, subject)
                          .replace(Mask.RELATION_MASK, relation))
        sro_texts = []
        if not self.test:
            for o in objects:
                entry = (self.prompt_template
                    .replace(Mask.SUBJECT_MASK, subject)
                    .replace(Mask.RELATION_MASK, relation)
                    .replace(Mask.OBJECT_MASK, o))
                sro_texts.append(entry)
        else:
            sro_texts.append(
                self.prompt_template
                    .replace(Mask.SUBJECT_MASK, subject)
                    .replace(Mask.RELATION_MASK, relation)
                    .replace(Mask.OBJECT_MASK, '')
            )

        x_coo = [self.triples[el] for el in reference]
        
        entities = set()
        relations = set()

        for s, r, o in x_coo:
            entities.add(s)
            relations.add(r)
            entities.add(o)
        
        entities = list(entities)
        relations = list(relations)
        
        
        x_coo = torch.tensor([[entities.index(t[0]), relations.index(t[1]), entities.index(t[2])] for t in x_coo])  # 0 - subject ; 1 - relation ; 2 - object

        entities = [self.id2entity[el] for el in entities]
        relations = [self.id2relation[el] for el in relations]

        x_coo = x_coo.T

        # text, entities, relations, x_coo, y_coo_cls

        return sr_graph_query, entities, relations, x_coo, sro_texts, objects

    # def process_prompt(self, subject, relation, objects):
    #     # return text_in, text_out
    #     text_in = apply_template(self.prompt_template, subject=subject, relation=relation, objects='')
    #     text_out = apply_template(self.prompt_template, subject=subject, relation=relation, objects=objects)
    #     return text_in, text_out
    
    # def process_graph_query(self, subject, relation):
    #     return apply_template(self.graph_query_template, subject=subject, relation=relation, objects=None)