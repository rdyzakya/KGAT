
from args import sg_train
import os
args = sg_train()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from kgat.model import (
    SubgraphGenerator,
    load_model_lmkbc
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

train_ds = SubgraphGenerationDataset(os.path.join(args.data, "train.json"), id2entity, id2rel)
val_ds = SubgraphGenerationDataset(os.path.join(args.data, "dev.json"), id2entity, id2rel)
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

### TRAIN
trainer = SubgraphGenerationTrainer(
    lmkbc_model=lmkbc_model,
    subgraph_generator=subgraphgenerator,
    tokenizer=tokenizer,
    train_ds=train_ds,
    val_ds=val_ds,
    test_ds=test_ds,
    epoch=args.epoch,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    last_hidden_state_bsize=args.hs_bsize,
    out_dir=args.out
)

train_history = trainer.train()

### EVALUATION
test_metrics = trainer.predict()

### SAVE MODEL, HISTORY, AND EVALUATION RESULT
trainer.save()