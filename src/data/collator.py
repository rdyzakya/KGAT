import torch

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def subgraphgen_collate_fn(batch):
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
    
    return text, entities, relations, x_coo, batch, y_coo_cls

def lmkbc_collate_fn(batch):
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

    return text_in, graph_query, entities, relations, x_coo, batch, text_out