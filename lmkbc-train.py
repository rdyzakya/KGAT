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

### PREPARE MODEL AND TOKENIZER
model_config = load_json(args.model)
model_name_or_path = model_config.pop("model_name_or_path")

lmkbc_model = load_model_lmkbc(model_name_or_path, checkpoint=None, device_map="auto", no_split_module_classes=['Block'])
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer = lmkbc_model.prepare_tokenizer(tokenizer)

### PREPARE DATASET
id2entity = load_id2map(os.path.join(args.data, "entities.txt"))
id2rel = load_id2map(os.path.join(args.data, "relations.txt"))
triples = load_json(os.path.join(args.data, "triples.json"))

train_ds = LMKBCDataset(os.path.join(args.data, "train.json"), 
                        id2entity, 
                        id2rel, 
                        triples, 
                        n_virtual_token=args.nvt,
                        n_data=args.n_data_train,
                        start_index=args.start_index_train,
                        eos_token=tokenizer.eos_token)

augment_ds = LMKBCDataset(os.path.join(args.data, "train.json"), 
                        id2entity, 
                        id2rel, 
                        triples, 
                        n_virtual_token=args.nvt,
                        n_data=args.n_data_train,
                        start_index=args.start_index_train,
                        test=True,
                        eos_token=tokenizer.eos_token)

val_ds = LMKBCDataset(os.path.join(args.data, "val.json"), 
                        id2entity, 
                        id2rel, 
                        triples, 
                        n_virtual_token=args.nvt,
                        n_data=args.n_data_val,
                        start_index=args.start_index_val,
                        eos_token=tokenizer.eos_token)
test_ds = LMKBCDataset(os.path.join(args.data, "test.json"), 
                        id2entity, 
                        id2rel, 
                        triples, 
                        n_virtual_token=args.nvt,
                        n_data=args.n_data_test,
                        start_index=args.start_index_test,
                        test=True,
                        eos_token=tokenizer.eos_token)



# LOAD MODEL
subgraphgenerator = SubgraphGenerator(
    n_features=lmkbc_model.embed_dim,
    **model_config
) if not args.from_sg else SubgraphGenerator.load(args.from_sg)


vt_generator = VirtualTokenGenerator.from_subgraph_generator(
    subgraphgenerator,
    n_virtual_token=args.nvt
) if not args.ckpt else VirtualTokenGenerator.load(args.ckpt)

pipeline = Pipeline(model=vt_generator, lmkbc_model=lmkbc_model)

### TRAIN
trainer1 = LMKBCTrainer(
    pipeline=pipeline,
    tokenizer=tokenizer,
    train_ds=train_ds if not args.no_train1 else None,
    val_ds=val_ds if not args.no_val1 else None,
    test_ds=augment_ds if not args.no_augment else None,
    epoch=args.epoch1,
    learning_rate=args.lr,
    batch_size=args.bsize,
    last_hidden_state_bsize=args.hsbsize,
    out_dir=os.path.join(args.out, "phase1"),
    max_check_point=args.mcp,
    best_metrics=args.best_metrics,
    load_best_model_at_end=False,
    optimizer=args.optim,
    logging_steps=args.logging_steps,
    beam_size=args.beam,
    max_length=args.max_length,
    optimizer_kwargs={},
)

if not args.no_train1:
    train_history1 = trainer1.train()

if not args.no_augment:
    test_metrics1, prediction_result1 = trainer1.predict()
    prediction_result1 = prediction_result1["lmkbc"]
    train_ds.augment(prediction_result1)


if not args.dont_save1:
    trainer1.save()

### AUGMENTTTT

trainer2 = LMKBCTrainer(
    pipeline=pipeline,
    tokenizer=tokenizer,
    train_ds=train_ds if not args.no_train2 else None,
    val_ds=val_ds if not args.no_val2 else None,
    test_ds=test_ds if not args.no_test else None,
    epoch=args.epoch2,
    learning_rate=args.lr,
    batch_size=args.bsize,
    last_hidden_state_bsize=args.hsbsize,
    out_dir=os.path.join(args.out, "phase2"),
    max_check_point=args.mcp,
    best_metrics=args.best_metrics,
    load_best_model_at_end=args.load_best,
    optimizer=args.optim,
    logging_steps=args.logging_steps,
    beam_size=args.beam,
    max_length=args.max_length,
    optimizer_kwargs={},
)

if not args.no_train2:
    train_history2 = trainer2.train()

if not args.no_val2:
    test_metrics2, prediction_result2 = trainer2.predict()

### SAVE MODEL, HISTORY, AND EVALUATION RESULT
if not args.dont_save2:
    trainer2.save()