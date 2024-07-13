
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

from seed import seed_everything

### RANDOM SEED
seed_everything(args.seed)

### PREPARE DATASET
id2entity = load_id2map(os.path.join(args.data, "entities.txt"))
id2rel = load_id2map(os.path.join(args.data, "relations.txt"))

train_ds = SubgraphGenerationDataset(os.path.join(args.data, "train.json"), id2entity, id2rel, n_data=args.n_data_train, split_size=args.split_size, start_index=args.start_index_train) if not args.no_train else None
val_ds = SubgraphGenerationDataset(os.path.join(args.data, "dev.json"), id2entity, id2rel, n_data=args.n_data_val, split_size=args.split_size, start_index=args.start_index_val) if not args.no_val else None
test_ds = SubgraphGenerationDataset(os.path.join(args.data, "test.json"), id2entity, id2rel, n_data=args.n_data_test, split_size=args.split_size, start_index=args.start_index_test) if not args.no_test else None

### PREPARE MODEL AND TOKENIZER
model_config = load_json(args.model)
model_name_or_path = model_config.pop("model_name_or_path")
# checkpoint = model_config.pop("checkpoint")

lmkbc_model = load_model_lmkbc(model_name_or_path, checkpoint=None, device_map="auto", no_split_module_classes=['Block'])
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer = lmkbc_model.prepare_tokenizer(tokenizer)


subgraphgenerator = SubgraphGenerator(
    n_features=lmkbc_model.embed_dim,
    **model_config
) if not args.ckpt else SubgraphGenerator.load(args.ckpt)

pipeline = Pipeline(model=subgraphgenerator, lmkbc_model=lmkbc_model)

### TRAIN
neg_loss_weight = args.nlw if args.nlw == "auto" else float(args.nlw)
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
    optimizer_kwargs={},
    neg_loss_weight=neg_loss_weight,
    alpha=args.alpha
)

if not args.no_train:
    train_history = trainer.train()

### EVALUATION
if not args.no_test:
    test_metrics = trainer.predict()

### SAVE MODEL, HISTORY, AND EVALUATION RESULT
trainer.save()