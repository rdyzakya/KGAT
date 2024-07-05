from args import lmkbc_train
import os
args = lmkbc_train()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from kgat.model import (
    VirtualTokenGenerator,
    SubgraphGenerator,
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

import torch

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
val_ds = LMKBCDataset(os.path.join(args.data, "val.json"), 
                        id2entity, 
                        id2rel, 
                        triples, 
                        prompt_template=args.pt,
                        graph_query_template=args.gqt,
                        n_virtual_token=args.nvt,
                        n_data=args.n_data_val)
test_ds = LMKBCDataset(os.path.join(args.data, "test.json"), 
                        id2entity, 
                        id2rel, 
                        triples, 
                        prompt_template=args.pt,
                        graph_query_template=args.gqt,
                        n_virtual_token=args.nvt,
                        n_data=args.n_data_test)

### PREPARE MODEL AND TOKENIZER
model_config = load_json(args.model)
model_name_or_path = model_config.pop("model_name_or_path")
checkpoint = model_config.pop("checkpoint")

lmkbc_model = load_model_lmkbc(model_name_or_path, checkpoint=checkpoint, device_map="auto", no_split_module_classes=['Block'])
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer = lmkbc_model.prepare_tokenizer(tokenizer)

subgraphgenerator = SubgraphGenerator(
    dim=lmkbc_model.embed_dim,
    **model_config
)

# LOAD MODEL
if args.from_sg:
    subgraphgenerator.load_state_dict(
        torch.load(args.from_sg)["state_dict"]
    )

vt_generator = VirtualTokenGenerator.from_subgraph_generator(
    subgraphgenerator,
    n_virtual_token=args.nvt
)

pipeline = Pipeline(model=vt_generator, lmkbc_model=lmkbc_model)

### TRAIN
trainer = LMKBCTrainer(
    pipeline=pipeline,
    tokenizer=tokenizer,
    train_ds=train_ds,
    val_ds=val_ds,
    test_ds=test_ds,
    epoch=args.epoch,
    learning_rate=args.lr,
    batch_size=args.bsize,
    last_hidden_state_bsize=args.hsbsize,
    out_dir=args.out,
    max_check_point=args.mcp,
    best_metrics=args.best_metrics,
    load_best_model_at_end=args.load_best,
    optimizer=args.optim,
    optimizer_kwargs={},
)

train_history = trainer.train()

### EVALUATION
test_metrics = trainer.predict()

### SAVE MODEL, HISTORY, AND EVALUATION RESULT
trainer.save()