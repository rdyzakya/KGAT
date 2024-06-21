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
        
        x_coo = torch.hstack(x_coo)
        batch = torch.tensor(batch)
        y_coo_cls = torch.hstack(y_coo_cls)

        # graph_query = tokenizer(batch[0], padding=True, truncation=True, max_length=64, return_tensors="pt")
        # entities = tokenizer(batch[1], padding=True, truncation=True, max_length=16, return_tensors="pt")
        # relations = tokenizer(batch[2], padding=True, truncation=True, max_length=16, return_tensors="pt")
        # x_coo = batch[3]
        # node_batch = batch[4]
        # y_coo = batch[5]

        # graph_module(graph_query_input_ids=graph_query["input_ids"], graph_query_attention_mask=graph_query["attention_mask"],
        # entities_input_ids=entities["input_ids"], entities_attention_mask=entities["attention_mask"],
        # relations_input_ids=relations["input_ids"], relations_attention_mask=relations["attention_mask"], 
        # x_coo=x_coo, batch=node_batch)

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
    
    # def pad_batch_input_ids(self, batch_input_ids):
    #     res = []
    #     index = [0]
    #     for input_ids in batch_input_ids:
    #         res.extend(input_ids.flip(-1) if self.left else input_ids)
    #         index.append(input_ids.shape[0] + index[-1])
    #     res = pad_sequence(res, batch_first=True, padding_value=self.tokenizer.pad_token_id)
    #     res = res.flip(-1) if self.left else res
    #     res = [res[index[i]:index[i+1]] for i in range(len(batch_input_ids))]
    #     return res

    # def pad_batch_attention_mask(self, batch_attention_masks):
    #     res = []
    #     index = [0]
    #     for attention_masks in batch_attention_masks:
    #         res.extend(attention_masks.flip(-1) if self.left else attention_masks)
    #         index.append(attention_masks.shape[0] + index[-1])
    #     res = pad_sequence(res, batch_first=True, padding_value=0)
    #     res = res.flip(-1) if self.left else res
    #     res = [res[index[i]:index[i+1]] for i in range(len(batch_attention_masks))]
    #     return res
    
    # def __call__(self, batch):
    #     pad_tokens = {
    #         "graph_query_input_ids" : self.tokenizer.pad_token_id, # SUCCESS
    #         "graph_query_attention_mask" : 0, # SUCCESS
    #         "entities_input_ids" : self.tokenizer.pad_token_id, # SUCCESS
    #         "entities_attention_mask" : 0, # SUCCESS
    #         "relations_input_ids" : self.tokenizer.pad_token_id, # SUCCESS
    #         "relations_attention_mask" : 0, # SUCCESS
    #         "x_coo" : -1, # SUCCESS
    #         "batch" : -1, # SUCCESS
    #         "y_coo_cls" : -1, # SUCCESS
    #     }
    #     n = max(1, int(np.ceil(len(batch) / self.n_process)))
        
    #     batch = [batch[i:i+n] for i in range(0,len(batch),n)]

    #     batch = [self.call_per_process(el) for el in batch]

    #     keys = pad_tokens.keys()
        
    #     res = {}
        
    #     for k in keys:
    #         batch_values = [el[k] for el in batch]
    #         if "input_ids" in k:
    #             batch_values = self.pad_batch_input_ids(batch_values)
    #         elif "attention_mask" in k:
    #             batch_values = self.pad_batch_attention_mask(batch_values)
    #         batch_values = pad_sequence(batch_values, batch_first=True, padding_value=pad_tokens[k])
    #         batch_values = batch_values.view(-1) if k == "batch" or k == "y_coo_cls" else batch_values.view(-1, batch_values.shape[-1])
    #         res[k] = batch_values
    #     return res


class LMKBCCollator:
    def __init__(self, tokenizer, n_process, left=True):
        self.tokenizer = tokenizer
        self.n_process = n_process
        self.left = left

    def __call__(self, batch):
        text_in, graph_query, entities, relations, x_coo, text_out = zip(*batch)

        text_in = list(text_in)
        graph_query = list(graph_query)

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
        # TODO bikin relation unik
        relations = flatten(relations)
        
        x_coo = torch.hstack(x_coo)
        batch = torch.tensor(batch)

        text_out = list(text_out)

        # prompt = tokenizer(batch[0], padding=True, truncation=True, max_length=64, return_tensors="pt")
        # graph_query = tokenizer(batch[1], padding=True, truncation=True, max_length=64, return_tensors=`"pt")
        # entities = tokenizer(batch[2], padding=True, truncation=True, max_length=16, return_tensors="pt")
        # relations = tokenizer(batch[3], padding=True, truncation=True, max_length=16, return_tensors="pt")
        # x_coo = batch[4]
        # node_batch = batch[5]
        # labels = tokenizer(batch[6], padding=True, truncation=True, max_length=64, return_tensors="pt")`

        # mean_fused_score, subgraph_emb, edge_batch = lmkbc_model.graph_module(
        #         graph_query["input_ids"], graph_query["attention_mask"],
        #         entities["input_ids"], entities["attention_mask"],
        #         relations["input_ids"], relations["attention_mask"],
        #         x_coo, node_batch)

        # out = lmkbc_model.text_module(prompt["input_ids"], prompt["attention_mask"], subgraph_emb)

        # def forward(self, graph_query_input_ids, graph_query_attention_mask,
        #         prompt_input_ids, prompt_attention_mask,
        #         entities_input_ids, entities_attention_mask,
        #         relations_input_ids, relations_attention_mask,
        #         x_coo, batch):

        graph_query = self.tokenizer(graph_query, padding=True, return_tensors="pt")
        prompt = self.tokenizer(text_in, padding=True, return_tensors="pt")
        entities = self.tokenizer(entities, padding=True, return_tensors="pt")
        relations = self.tokenizer(relations, padding=True, return_tensors="pt")
        labels = self.tokenizer(text_out, padding=True, return_tensors="pt")

        # return text_in, graph_query, entities, relations, x_coo, batch, text_out
        return {
            "graph_query_input_ids" : graph_query["input_ids"],
            "graph_query_attention_mask" : graph_query["attention_mask"],
            "prompt_input_ids" : prompt["input_ids"],
            "prompt_attention_mask" : prompt["attention_mask"],
            "entities_input_ids" : entities["input_ids"],
            "entities_attention_mask" : entities["attention_mask"],
            "relations_input_ids" : relations["input_ids"],
            "relations_attention_mask" : relations["attention_mask"],
            "x_coo" : x_coo,
            "batch" : batch,
            "labels" : labels["input_ids"]
        }