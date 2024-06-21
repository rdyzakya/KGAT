
from args import sg_train
import os
args = sg_train()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from kgat.model import (
    SubgraphGenerator,
    load_model_lmkbc,
    Pipeline
)

from kgat.data import (
    SubgraphGenerationDataset,
    load_id2map,
    load_json
)

from kgat.trainer import SubgraphGenerationTrainer

from transformers import (
    AutoTokenizer
)

### PREPARE DATASET
id2entity = load_id2map(os.path.join(args.data, "entities.txt"))
id2rel = load_id2map(os.path.join(args.data, "relations.txt"))

train_ds = SubgraphGenerationDataset(os.path.join(args.data, "train.json"), id2entity, id2rel, n_data=args.n_data_train)
val_ds = SubgraphGenerationDataset(os.path.join(args.data, "dev.json"), id2entity, id2rel, n_data=args.n_data_val)
test_ds = SubgraphGenerationDataset(os.path.join(args.data, "test.json"), id2entity, id2rel)

### PREPARE MODEL AND TOKENIZER
model_config = load_json(args.model)
model_name_or_path = model_config.pop("model_name_or_path")

lmkbc_model = load_model_lmkbc(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer = lmkbc_model.prepare_tokenizer(tokenizer)


subgraphgenerator = SubgraphGenerator(
    input_dim=lmkbc_model.embed_dim,
    **model_config
)

pipeline = Pipeline(model=subgraphgenerator, lmkbc_model=lmkbc_model)

### TRAIN
trainer = SubgraphGenerationTrainer(
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
    optimizer_kwargs={}
)

train_history = trainer.train()

### EVALUATION
test_metrics = trainer.predict()

### SAVE MODEL, HISTORY, AND EVALUATION RESULT
trainer.save()