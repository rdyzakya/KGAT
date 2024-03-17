# LM-KBC Dataset

## Source
1. lm-kbc2022 : https://github.com/lm-kbc/dataset2022
2. lm-kbc2023 : https://github.com/lm-kbc/dataset2023
3. mars : https://github.com/zjunlp/MKG_Analogy
4. freebase : https://paperswithcode.com/dataset/fb15k-237 & https://github.com/xiaoling/figer/issues/6 or https://download.csdn.net/download/guotong1988/9865898 (mid2name)
5. conceptnet : https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_release.zip (QAGNN)

## Convention
1. Each dataset from each source will be splitted in to train (train.json), validation (val.json), and test (test.json)
2. Each json file will contain keys such as:
    * num_triplets : number of triplets
    * num_entities : number of entities
    * num_relations : number of relations
    * entity : entity mapping, the key is the index, the value is the list of aliases for the corresponding entity
    * relation : relation mapping, the key is the index, the value is the relation name
    * coo : coordinate list, list of triplets, the order for each triplet is [subject index, object index, relation index] or [head index, tail index, relation index]