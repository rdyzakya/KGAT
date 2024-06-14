from args import train_sg_args
import os
args = train_sg_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from kgat import (load_json,
                  load_id2map,
                  SubgraphGenerationDataset,
                  SubgraphGenerationCollator,
                  SGTrainer,
                  load_config_sg)
from transformers import AutoTokenizer, TrainingArguments
import evaluate

# from torch.distributed.fsdp import (
#    FullyShardedDataParallel,
#    CPUOffload,
# )
# from torch.distributed.fsdp.wrap import (
#    default_auto_wrap_policy,
# )
# from torch.nn.parallel import (
#     DistributedDataParallel
# )

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
    model_config = load_json(args.model)

    # PREPARE DATASET
    train_ds, val_ds, test_ds = prepare_data(args.data)

    # PREPARE MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_config["clm"]["model_name_or_path"], padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = load_config_sg(model_config, clm=None)
    model.transformer.config.pad_token_id = tokenizer.eos_token_id


    # PREPARE TRAINER
    n_process = len(args.gpu.split(','))
    sg_collator = SubgraphGenerationCollator(tokenizer=tokenizer, n_process=n_process)

    # f1_score = evaluate.load("f1")
    # accuracy = evaluate.load("accuracy")

    # def compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     # predictions = np.argmax(predictions, axis=1)
    #     predictions = predictions.argmax(-1)
    #     f1 = f1_score.compute(predictions=predictions, references=labels, average="macro")
    #     acc = accuracy.compute(predictions=predictions, references=labels)

    #     res = {
    #         "f1" : f1["f1"],
    #         "accuracy" : acc["accuracy"]
    #     }
    #     return res

    model.freeze_llm()

    # model = DistributedDataParallel(model)
    # model = FullyShardedDataParallel(
    # model(),
    # fsdp_auto_wrap_policy=default_auto_wrap_policy,
    # cpu_offload=CPUOffload(offload_params=True),
    # )

    train_args = TrainingArguments(**train_config)
    trainer = SGTrainer(
        model=model,
        args=train_args,
        data_collator=sg_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    ) # argsss

    trainer.train()


    # if trainer.is_fsdp_enabled:
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(train_config.output_dir)


if __name__ == "__main__":
    main(args)