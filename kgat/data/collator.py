import torch

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


class SubgraphGenerationCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

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

        return {
            "graph_query_input_ids" : graph_query["input_ids"],
            "graph_query_attention_mask" : graph_query["attention_mask"],
            "entities_input_ids" : entities["input_ids"],
            "entities_attention_mask" : entities["attention_mask"],
            "relations_input_ids" : relations["input_ids"],
            "relations_attention_mask" : relations["attention_mask"],
            "x_coo" : x_coo,
            "batch" : batch,
            "y_coo_cls" : y_coo_cls
        }
    
class LMKBCCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

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