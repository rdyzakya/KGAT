# Subgraph Generation Dataset

## Source

1. atomic : https://allenai.org/data/atomic
2. qagnn : https://github.com/michiyasunaga/qagnn
3. text2kg : https://github.com/cenguix/Text2KGBench
4. webnlg : https://huggingface.co/datasets/web_nlg
5. graph-writer : https://github.com/rikdz/GraphWriter

## Convention

1. Each dataset from each source will be splitted in to train (train.json), validation (val.json), and (maybe) test (test.json)
2. There will be 2 txt files, entities.txt and relations.txt
3. Each json file will contain list of entries containing keys such as:
    * text : text
    * entities : list of entity index (corresponds to entities.txt)
    * relations : list of relation index (corresponds to relations.txt)
    * x_coo : list of coo triples (entity and relation index), for each triple contains [subject id, relation id, object id]
    * y_coo_cls : correspond to the x_coo (1 if the triples exist, 0 otherwise)
    * y_node_cls : list of node labels that corresponds to entities (1 if the node is related with the text, 0 otherwise)

## TODO
1. add edge cases where there are no subgraph that relates