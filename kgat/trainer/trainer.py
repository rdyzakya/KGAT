import math
from tqdm import tqdm
import os
import torch
import numpy as np
from abc import ABC
import json
import shutil
import re
from .config import Config
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch import optim

optimizer_catalog = {
    "sgd" : optim.SGD,
    "adam" : optim.Adam
}

def get_optimizer(name):
    return optimizer_catalog[name]

class Trainer(ABC):
    def __init__(self, 
                 pipeline,
                 tokenizer,
                 train_ds,
                 val_ds=None,
                 test_ds=None,
                 epoch=10, 
                 learning_rate=1e-3, 
                 batch_size=4,
                 last_hidden_state_bsize=16,
                 out_dir="./out",
                 max_check_point=3,
                 best_metrics="loss",
                 load_best_model_at_end=False,
                 optimizer="sgd",
                 optimizer_kwargs={},
                 **kwargs):
        
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        self.val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn) if val_ds else None
        self.test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn) if test_ds else None
        self.accelerator = Accelerator()

        config_kwargs = {k : v for k, v in kwargs.items() if self.__is_config_args(v)}
        self.config = Config(epoch=epoch,
                             learning_rate=learning_rate,
                             batch_size=batch_size,
                             last_hidden_state_bsize=last_hidden_state_bsize,
                             out_dir=out_dir,
                             max_check_point=max_check_point,
                             best_metrics=best_metrics,
                             load_best_model_at_end=load_best_model_at_end,
                             optimizer=optimizer,
                             **config_kwargs)
        self.history = []
        self.test_metrics = {}
        self.prediction_result = [] # list of dicts
        self.optimizer = get_optimizer(optimizer)(self.pipeline.model.parameters(), lr=learning_rate, **optimizer_kwargs)
    
    def __is_config_args(self, value):
        return isinstance(value, int) or isinstance(value, float) or isinstance(value, str)
    
    def criterion(self, preds, labels):
        raise NotImplementedError("Abstrack class")
    
    def run_epoch(self, bar, train=False):
        raise NotImplementedError("Abstract class")
    
    def compute_metrics(self, preds, labels, prefix=None):
        raise NotImplementedError("Abstract class")
    
    def architecture(self, unwrapped_model):
        raise NotImplementedError("Abstract class")
    
    def log(self, text):
        if self.accelerator.is_main_process:
            print(text)
    
    def prepare_train(self):
        (self.pipeline.model,
         self.train_dataloader, 
         self.val_dataloader, 
         self.test_dataloader, 
         self.optimizer) = self.accelerator.prepare(
            self.pipeline.model,
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
            self.optimizer
        )
        self.lmkbc_model_device = self.pipeline.lmkbc_model.device
        self.model_device = self.accelerator.device
    
    def train(self):
        self.log(f"You are training using {torch.cuda.device_count()} GPU(s)")
        self.history = []
        train_steps_per_epoch = math.ceil(len(self.train_dataloader.dataset) / self.config.batch_size)
        train_steps = train_steps_per_epoch * self.config.epoch
        train_bar = tqdm(total=train_steps, desc="Training")

        os.makedirs(self.config.out_dir, exist_ok=True)
        checkpoints = [el for el in os.listdir(self.config.out_dir) if el.startswith("checkpoint-") and os.path.isdir(os.path.join(self.config.out_dir, el)) ]
        if len(checkpoints) > 0:
            last_checkpoint = max(checkpoints, key=lambda x: int(x.replace("checkpoint-",'')))
            last_epoch = int(last_checkpoint.replace("checkpoint-",''))
            self.log(f"Resume training on epoch {last_epoch+1}")
            self.pipeline.model.load_state_dict(
                torch.load(os.path.join(self.config.out_dir, last_checkpoint, "model.pth"))["state_dict"]
            )
            train_bar.update(train_steps * last_epoch)
        
        self.prepare_train()

        for e in range(self.config.epoch):
            train_metrics = self.run_epoch(self.train_dataloader, train_bar, train=True)
            entry = {
                "epoch" : e+1
            }
            entry.update(train_metrics)
            if self.val_dataloader:
                val_steps = math.ceil(len(self.val_dataloader.dataset) / self.config.batch_size)
                val_bar = tqdm(total=val_steps, desc=f"Evaluation epoch {e+1}")
                val_metrics = self.run_epoch(self.val_dataloader, val_bar, train=False)
                entry.update(val_metrics)
            self.log(entry)
            self.history.append(entry)
            self.update_check_point()
        
        # Load best model at end
        if self.config.load_best_model_at_end and self.val_dataloader:
            metrics_history = [el[f"val_{self.config.best_metrics}"] for el in self.history]
            best_epoch = np.argmin(metrics_history) if self.config.best_metrics.endswith("loss") else np.argmax(metrics_history)
            best_checkpoint = f"checkpoint-{best_epoch}"
            self.pipeline.model.load_state_dict(
                torch.load(os.path.join(self.config.out_dir, best_checkpoint, "model.pth"))["state_dict"]
            )
        return self.history
    

    def predict(self, test_dataloader=None):
        raise NotImplementedError("Abstract class")
    
    def save(self, directory=None, save_history=True, save_evaluation_metrics=True, save_train_config=True, save_prediction_result=True):
        # TODO : save prediction result
        directory = directory or self.config.out_dir
        # Make directory
        os.makedirs(directory, exist_ok=True)
        if save_history:
            # Save history
            with open(os.path.join(directory, "history.json"), 'w') as fp:
                json.dump(self.history, fp)
        if save_evaluation_metrics:
            # Save test metrics
            with open(os.path.join(directory, "evaluation_metrics.json"), 'w') as fp:
                json.dump(self.test_metrics, fp)
        if save_train_config:
            # Save train config
            with open(os.path.join(directory, "train_config.json"), 'w') as fp:
                json.dump(self.config.to_dict(), fp)
        if save_prediction_result:
            # Save prediction resukt
            with open(os.path.join(directory, "preds.json"), 'w') as fp:
                json.dump(self.prediction_result, fp)

        # Save model
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.pipeline.model)
        torch.save({
            "state_dict" : unwrapped_model.state_dict(),
            "architecture" : self.architecture(unwrapped_model)
        }, os.path.join(directory, "model.pth"))
    
    def update_check_point(self):
        if len(self.history) > self.config.max_check_point:
            best_checkpoint = None
            if self.val_dataloader:
                metrics_history = [el[f"val_{self.config.best_metrics}"] for el in self.history]

                best_index = np.argmin(metrics_history) if self.config.best_metrics.endswith("loss") else np.argmax(metrics_history)
                best_checkpoint = f"checkpoint-{best_index}"
            # Delete Check point if exceed limit
            checkpoints = [el for el in os.listdir(self.config.out_dir) if re.match(r"checkpoint-\d+", el)
                        and os.path.isdir(os.path.join(self.config.out_dir, el)) and el != best_checkpoint]
            deleted_checkpoint = min(checkpoints, key=lambda x: int(x.replace("checkpoint-", '')))
            shutil.rmtree(os.path.join(self.config.out_dir, deleted_checkpoint))
        
        # Save model
        current_epoch = len(self.history) - 1
        self.save(directory=os.path.join(self.config.out_dir, f"checkpoint-{current_epoch}"),
                  save_history=True,
                  save_evaluation_metrics=False,
                  save_train_config=False)