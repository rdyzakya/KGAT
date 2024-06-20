from args import train_sg_args
import os
args = train_sg_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from kgat import (load_json,
                  load_id2map,
                  SubgraphGenerationDataset,
                  load_config_sg,
                  SGTrainer)
from transformers import AutoTokenizer
import torch
import json

def prepare_data(data_dir):
    id2entity = load_id2map(os.path.join(data_dir, "entities.txt"))
    id2rel = load_id2map(os.path.join(data_dir, "relations.txt"))

    train_ds = SubgraphGenerationDataset(os.path.join(data_dir, "train.json"), id2entity, id2rel)
    val_ds = SubgraphGenerationDataset(os.path.join(data_dir, "dev.json"), id2entity, id2rel)
    test_ds = SubgraphGenerationDataset(os.path.join(data_dir, "test.json"), id2entity, id2rel)

    return train_ds, val_ds, test_ds

# SHOULD USE SOMETHING LIKE PEFT
def main(args):

    train_config = load_json(args.train)
    # model_config = load_json(args.model)
    model_config = torch.load(args.model) if args.model.endswith(".pth") else load_json(args.model)
    # PREPARE DATASET
    train_ds, val_ds, test_ds = prepare_data(args.data)

    # PREPARE MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_config["structure"]["clm"]["model_name_or_path"], padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = load_config_sg(model_config, clm=None)
    model.transformer.config.pad_token_id = tokenizer.eos_token_id
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.freeze_llm()

    # PREPARE TRAINER
    trainer = SGTrainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        train_batch_size=train_config["per_device_train_batch_size"],
        val_batch_size=train_config["per_device_eval_batch_size"],
        epoch=train_config["num_train_epochs"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"]
    )

    model, history = trainer.train_loop()

    os.makedirs(train_config["output_dir"], exist_ok=True)
    trainer.save_model(model_config, out_path=os.path.join(train_config["output_dir"], "model.pth"))
    with open(os.path.join(train_config["output_dir"], "history.json"), 'w') as fp:
        json.dump(history, fp)



if __name__ == "__main__":
    main(args)