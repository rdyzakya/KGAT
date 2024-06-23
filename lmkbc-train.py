from args import lmkbc_train
import os
args = lmkbc_train()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from kgat.model import (
    VirtualTokenGenerator,
    load_model_lmkbc,
    Pipeline
)

from kgat.data import (
    LMKBCDataset,
    load_id2map,
    load_json
)

from kgat.trainer import LMKBCTrainer

from transformers import (
    AutoTokenizer
)

from seed import seed_everything

### RANDOM SEED
seed_everything(args.seed)

### PREPARE DATASET
id2entity = load_id2map(os.path.join(args.data, "entities.txt"))
id2rel = load_id2map(os.path.join(args.data, "relations.txt"))
triples = load_json(os.path.join(args.data, "triples.json"))

train_ds = LMKBCDataset(os.path.join(args.data, "train.json"), 
                        id2entity, 
                        id2rel, 
                        triples, 
                        prompt_template=args.pt,
                        graph_query_template=args.gqt,
                        n_virtual_token=args.nvt,
                        n_data=args.n_data_train)
val_ds = LMKBCDataset(os.path.join(args.data, "dev.json"), id2entity, id2rel, n_data=args.n_data_val)
test_ds = LMKBCDataset(os.path.join(args.data, "test.json"), id2entity, id2rel)