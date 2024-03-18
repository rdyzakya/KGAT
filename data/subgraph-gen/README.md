# Subgraph Generation Dataset

## Source

1. atomic : https://allenai.org/data/atomic
2. qagnn : https://github.com/michiyasunaga/qagnn
3. text2kg : https://github.com/cenguix/Text2KGBench
4. webnlg : https://huggingface.co/datasets/web_nlg
5. graphwriter : https://github.com/rikdz/GraphWriter

## Convention

1. Each dataset from each source will be splitted in to train (train.json), validation (val.json), and (maybe) test (test.json)
2. There will be 2 txt files, entities.txt and relations.txt
3. Each json file will contain list of entries containing keys such as:
    * text : text
    * entities : list of entity index (corresponds to entities.txt)
    * relations : list of relation index (corresponds to relations.txt)
    * x_coo : input coo with shape of (3, TX) with TX is the number of triplets in x_coo
    * y_coo : output coo with shape of (3, TY) with TY is the number of triplets in y_coo, the triplets that is related with the text
    * y_node_cls : list of node labels that corresponds to entities (0 if the node is not related with the text, 1 otherwise)