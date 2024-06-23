import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

class SubgraphGenerationCollator:
    def __init__(self, tokenizer, n_process, left=True):
        self.tokenizer = tokenizer
        self.n_process = n_process
        self.left = left

    def __call__(self, batch):
        text, entities, relations, x_coo, y_coo_cls = zip(*batch)

        text = list(text)

        batch = []

        for i, e in enumerate(entities):
            batch.extend([i for _ in range(len(e))])
        
        prev_len_entity = 0
        prev_len_relation = 0
        for i in range(len(x_coo)):
            x_coo[i][0] += prev_len_entity
            x_coo[i][1] += prev_len_relation
            x_coo[i][2] += prev_len_entity
            
            prev_len_entity += len(entities[i])
            prev_len_relation += len(relations[i])
        
        entities = flatten(entities)
        relations = flatten(relations)

        new_relations = sorted(set(relations))
        new_relations_id_map = torch.tensor([new_relations.index(el) for el in relations])

        relations = new_relations

        x_coo = torch.hstack(x_coo)
        x_coo[1,:] = new_relations_id_map[x_coo[1,:]]
        batch = torch.tensor(batch)
        y_coo_cls = torch.hstack(y_coo_cls)

        graph_query = self.tokenizer(text, padding=True, return_tensors="pt")
        entities = self.tokenizer(entities, padding=True, return_tensors="pt")
        relations = self.tokenizer(relations, padding=True, return_tensors="pt")

        x_coo = x_coo.transpose(0,1)

        return {
            "graph_query_input_ids" : graph_query["input_ids"], # N_query, length
            "graph_query_attention_mask" : graph_query["attention_mask"], # N_query, length
            "entities_input_ids" : entities["input_ids"], # N_entities, length
            "entities_attention_mask" : entities["attention_mask"], # N_entities, length
            "relations_input_ids" : relations["input_ids"], # N_relations, length
            "relations_attention_mask" : relations["attention_mask"], # N_relations, length
            "x_coo" : x_coo, # N_triplets, 3
            "batch" : batch, # N_entities
            "y_coo_cls" : y_coo_cls # N_triplets
        }


class LMKBCCollator:
    def __init__(self, tokenizer, n_process, left=True, test=False):
        self.tokenizer = tokenizer
        self.n_process = n_process
        self.left = left
        self.test = test

    def __call__(self, batch):
        # text_in, graph_query, entities, relations, x_coo, text_out = zip(*batch)
        sr_graph_query, entities, relations, x_coo, sro_texts, objects = zip(*batch)

        sr_graph_query = list(sr_graph_query)

        entities_batch = []

        for i, e in enumerate(entities):
            entities_batch.extend([i for _ in range(len(e))])
        
        graph_emb_batch = []
        for i, sro in enumerate(sro_texts):
            graph_emb_batch.extend([i for _ in range(len(sro))])
        
        prev_len_entity = 0
        prev_len_relation = 0
        for i in range(len(x_coo)):
            x_coo[i][0] += prev_len_entity
            x_coo[i][1] += prev_len_relation
            x_coo[i][2] += prev_len_entity
            
            prev_len_entity += len(entities[i])
            prev_len_relation += len(relations[i])
        
        entities = flatten(entities)
        relations = flatten(relations)

        new_relations = sorted(set(relations))
        new_relations_id_map = torch.tensor([new_relations.index(el) for el in relations])

        relations = new_relations
        
        x_coo = torch.hstack(x_coo)
        x_coo[1,:] = new_relations_id_map[x_coo[1,:]]
        entities_batch = torch.tensor(entities_batch)
        graph_emb_batch = torch.tensor(graph_emb_batch)

        graph_query = self.tokenizer(sr_graph_query, padding=True, return_tensors="pt")
        entities = self.tokenizer(entities, padding=True, return_tensors="pt")
        relations = self.tokenizer(relations, padding=True, return_tensors="pt")

        x_coo = x_coo.transpose(0,1)

        lmkbc_text = flatten(sro_texts)
        lmkbc_text = [el + self.tokenizer.eos_token for el in lmkbc_text] if not self.test else lmkbc_text
        lmkbc_text = self.tokenizer(lmkbc_text, padding=True, return_tensors="pt")

        lmkbc_input_ids = lmkbc_text["input_ids"]
        lmkbc_attention_mask = lmkbc_text["attention_mask"]
        lmkbc_labels = lmkbc_input_ids.clone()
        lmkbc_labels[lmkbc_attention_mask == 0] = -100
        lmkbc_labels[lmkbc_labels == self.tokenizer.kg_token_id] = -100 # accustomed to n_virtual_token


        #  shift_logits = lm_logits[..., :-1, :].contiguous()
        # shift_labels = labels[..., 1:].contiguous()

        n_object = torch.tensor([len(o) for o in objects])

        return {
            "graph_query_input_ids" : graph_query["input_ids"], # N_query, length
            "graph_query_attention_mask" : graph_query["attention_mask"], # N_query, length
            "entities_input_ids" : entities["input_ids"], # N_entities, length
            "entities_attention_mask" : entities["attention_mask"], # N_entities, length
            "relations_input_ids" : relations["input_ids"], # N_relations, length
            "relations_attention_mask" : relations["attention_mask"], # N_relations, length
            "x_coo" : x_coo, # N_triplets, 3
            "entities_batch" : entities_batch, # N_entities
            "lmkbc_input_ids" : lmkbc_input_ids,
            "lmkbc_attention_mask" : lmkbc_attention_mask,
            "lmkbc_labels" : lmkbc_labels,
            "graph_emb_batch" : graph_emb_batch, # N_lmkbc_text
            "objects" : flatten(objects),
            "n_object" : n_object
            # TODO weights?
        }