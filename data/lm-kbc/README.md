# LM-KBC Dataset

## Source
1. lm-kbc2022 : https://github.com/lm-kbc/dataset2022
2. lm-kbc2023 : https://github.com/lm-kbc/dataset2023
3. mars : https://github.com/zjunlp/MKG_Analogy
4. freebase : https://paperswithcode.com/dataset/fb15k-237 & https://github.com/xiaoling/figer/issues/6 or https://download.csdn.net/download/guotong1988/9865898 (mid2name)
5. conceptnet : https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_release.zip (QAGNN)

## Convention
1. Each dataset from each source will be splitted in to train (train.json), validation (val.json), and test (test.json)
2. Each dataset contains several additional files such as:
    * entities.txt : list of entities
    * relations.txt : list of relations
    * triples.json : list of triples in the whole knowledge graph, the triple format is [subject id, object id, relation id]
2. Each split json file will contain list of records with keys such as:
    * subject : subject id, subject alias can be found in entities.txt
    * relation : relation id, relation alias can be found in relations.txt
    * objects : list of object ids (may be empty, only 1, more than 1), object alias can be found in entities.txt
    * reference : list of triple index, the triple can be found in coo.json