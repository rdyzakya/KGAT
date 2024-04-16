from argparse import ArgumentParser

import torch.utils
from kgat import (load_config_sg,
                  load_json,
                  load_id2map,
                  SubgraphGenerationDataset,
                  subgraphgen_collate_fn)
import torch
import os

def init_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="model config json path", 
                        default="./config/model/default.json")
    parser.add_argument("-t", "--train", type=str, help="train config json path",
                        default="./config/train/sg-default.json")
    args = parser.parse_args()
    return args

def main():
    args = init_args()

    train_config = load_json(args.train)

    # PREPARE DATASET
    train_ds, val_ds, test_ds = prepare_data(train_config["data_dir"])
    # PREPARE DATALOADER
    train_dataloader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=train_config["train_batch_per_device"],
                                                   # should be distribute accros devices
                                                   shuffle=train_config["shuffle"],
                                                   collate_fn=subgraphgen_collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_ds,
                                                batch_size=train_config["val_batch_per_device"],
                                                # should be distribute accros devices
                                                shuffle=False,
                                                collate_fn=subgraphgen_collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_ds,
                                                batch_size=train_config["test_batch_per_device"],
                                                # should be distribute accros devices
                                                shuffle=False,
                                                collate_fn=subgraphgen_collate_fn)
    # PREPARE MODEL
    model_config = load_json(args.model)
    graph_module = load_config_sg(model_config)
    # TRAIN

def prepare_data(data_dir):
    id2entity = load_id2map(os.path.join(data_dir, "entities.txt"))
    id2rel = load_id2map(os.path.join(data_dir, "relations.txt"))

    train_ds = SubgraphGenerationDataset(os.path.join(data_dir, "train.json"), id2entity, id2rel)
    val_ds = SubgraphGenerationDataset(os.path.join(data_dir, "val.json"), id2entity, id2rel)
    test_ds = SubgraphGenerationDataset(os.path.join(data_dir, "test.json"), id2entity, id2rel)

    return train_ds, val_ds, test_ds