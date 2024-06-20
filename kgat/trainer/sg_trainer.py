from torch.optim import SGD
from torch.nn import BCEWithLogitsLoss
from accelerate import Accelerator
from torch.utils.data import DataLoader
from ..data import SubgraphGenerationCollator
from tqdm import tqdm
from sklearn.metrics import classification_report
import torch
import utils
import time
import math
import os
import json
import re
import numpy as np
import shutil

class SubgraphGenerationTrainer:
    def __init__(self, 
                 lmkbc_model, 
                 subgraph_generator, 
                 tokenizer,
                 train_ds,
                 alpha=1.0,
                 val_ds=None,
                 test_ds=None,
                 epoch=10, 
                 learning_rate=1e-3, 
                 batch_size=4,
                 last_hidden_state_bsize=16,
                 out_dir="./out",
                 max_check_point=3,
                 best_metrics="loss",
                 load_best_model_at_end=False):
        
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.last_hidden_state_bsize = last_hidden_state_bsize
        self.alpha = alpha
        self.out_dir = out_dir
        self.history = []
        self.test_metrics = {}
        self.max_check_point = max_check_point
        self.best_metrics = best_metrics
        self.load_best_model_at_end = load_best_model_at_end

        self.collate_fn = SubgraphGenerationCollator(tokenizer=tokenizer, 
                                                     n_process=torch.cuda.device_count(), 
                                                     left=True)
        self.criterion = BCEWithLogitsLoss()
        self.accelerator = Accelerator()

        self.lmkbc_model = lmkbc_model
        self.subgraph_generator = subgraph_generator
        self.train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        self.val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None
        self.test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds else None
        self.optimizer = SGD(subgraph_generator.parameters(), lr=learning_rate)
    
    def prepare_train(self):
        (self.lmkbc_model, 
         self.subgraph_generator, 
         self.train_dataloader, 
         self.val_dataloader, 
         self.test_dataloader, 
         self.optimizer) = self.accelerator.prepare(
            self.lmkbc_model,
            self.subgraph_generator,
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
            self.optimizer
        )

    def create_score_matrix(self, n_entities, n_relations, x_coo, y_coo_cls=None):
        score_matrix = torch.zeros(n_entities, n_relations, n_entities, dtype=torch.float32)
        x_coo = x_coo[y_coo_cls.bool()] if y_coo_cls else x_coo
        for el in x_coo:
            score_matrix[el[0], el[1], el[2]] = 1.0
        return score_matrix

    def compute_metrics(self, preds, labels, prefix=None):
        prefix = prefix or ""

        report = classification_report(y_true=labels, y_pred=preds)
        return {
            f"{prefix}accuracy" : report["accuracy"],
            f"{prefix}precision" : report["weighted avg"]["precision"],
            f"{prefix}recall" : report["weighted avg"]["recall"],
            f"{prefix}f1" : report["weighted avg"]["f1-score"],
        }

    def run_epoch(self, dataloader, bar, train=True):
        if train:
            self.lmkbc_model.train()
            self.subgraph_generator.train()
        else:
            self.lmkbc_model.eval()
            self.subgraph_generator.eval()

        loss_data = torch.zeros(2, dtype=torch.float32)

        all_sg_preds = []
        all_sg_labels = []

        all_gg_preds = []
        all_gg_labels = []

        start_time = time.time()

        for batch in dataloader:
            if train:
                self.optimizer.zero_grad()
            
            with utils.context_manager(train=train):
                queries = self.lmkbc_model.batch_last_hidden_state(
                    input_ids=batch["graph_query_input_ids"],
                    attention_mask=batch["graph_query_attention_mask"],
                    batch_size=self.last_hidden_state_bsize
                )

                entities = self.lmkbc_model.batch_last_hidden_state(
                    input_ids=batch["entities_input_ids"],
                    attention_mask=batch["entities_attention_mask"],
                    batch_size=self.last_hidden_state_bsize
                )

                relations = self.lmkbc_model.batch_last_hidden_state(
                    input_ids=batch["relations_input_ids"],
                    attention_mask=batch["relations_attention_mask"],
                    batch_size=self.last_hidden_state_bsize
                )

                sg_out = self.subgraph_generator(
                    queries=queries,
                    entities=entities,
                    relations=relations,
                    x_coo=batch["x_coo"],
                    batch=batch["batch"]
                )

                sg_labels = self.create_score_matrix(
                    n_entities=entities.shape[0],
                    n_relations=relations.shape[0],
                    x_coo=batch["x_coo"],
                    y_coo_cls=batch["y_coo_cls"]
                )

                sg_loss = self.criterion(sg_out.view(-1), sg_labels.view(-1))
                
                all_sg_preds.append(sg_out.view(-1).sigmoid().round().int())
                all_sg_labels.append(sg_labels.view(-1).int())

                loss = self.alpha * sg_loss

                if self.alpha < 1.0:
                    gg_out = self.subgraph_generator.encoder_decoder(
                        entities=entities,
                        relations=relations,
                        x_coo=batch["x_coo"]
                    )

                    gg_labels = self.create_score_matrix(
                        n_entities=entities.shape[0],
                        n_relations=relations.shape[0],
                        x_coo=batch["x_coo"],
                        y_coo_cls=None
                    )

                    gg_loss = self.criterion(gg_out.view(-1), gg_labels.view(-1))

                    all_gg_preds.append(gg_out.view(-1).sigmoid().round().int())
                    all_gg_labels.append(gg_labels.view(-1).int())

                    loss += (1 - self.alpha) * gg_loss
            if train:
                self.accelerator.backward(loss)
                self.optimizer.step()
            loss_data[0] += loss
            loss_data[1] += (sg_out.shape[0] * sg_out.shape[1] * sg_out.shape[2])

            bar.update(1)
        
        end_time = time.time()

        total_loss = loss_data[0] / loss_data[1]

        all_sg_preds = torch.cat(all_sg_preds)
        all_sg_labels = torch.cat(all_sg_labels)

        all_gg_preds = torch.cat(all_gg_preds)
        all_gg_labels = torch.cat(all_gg_labels)

        prefix = "train_" if train else "val_"

        metrics = {
            f"{prefix}time" : end_time - start_time,
            f"{prefix}loss" : total_loss.item()
        }
        metrics.update(
            self.compute_metrics(all_sg_preds, all_sg_labels, prefix=f"{prefix}sg_")
        )
        if self.alpha < 1.0:
            metrics.update(
                self.compute_metrics(all_gg_preds, all_gg_labels, prefix=f"{prefix}gg_")
            )
        return metrics

    def train(self):
        self.history = []
        train_steps_per_epoch = math.ceil(len(self.train_dataloader.dataset) / self.batch_size)
        train_steps = train_steps_per_epoch * self.epoch
        train_bar = tqdm(total=train_steps, desc="Training")

        os.makedirs(self.out_dir, exist_ok=True)
        checkpoints = [el for el in os.listdir(self.out_dir) if el.startswith("checkpoint-") and os.path.isdir(os.path.join(self.out_dir, el)) ]
        if len(checkpoints) > 0:
            last_checkpoint = max(checkpoints, key=lambda x: int(x.replace("checkpoint-",'')))
            last_epoch = int(last_checkpoint.replace("checkpoint-",''))
            print(f"Resume training on epoch {last_epoch+1}")
            self.subgraph_generator.load_state_dict(
                torch.load(os.path.join(self.out_dir, last_checkpoint, "model.pth"))
            )
            train_bar.update(train_steps * last_epoch)
        
        self.prepare_train()

        for e in range(self.epoch):
            train_metrics = self.run_epoch(self.train_dataloader, train_bar, train=True)
            entry = {
                "epoch" : e+1
            }
            entry.update(train_metrics)
            if self.val_dataloader:
                val_steps = math.ceil(len(self.val_dataloader.dataset) / self.batch_size)
                val_bar = tqdm(total=val_steps, desc=f"Evaluation epoch {e+1}")
                val_metrics = self.run_epoch(self.val_dataloader, val_bar, train=False)
                entry.update(val_metrics)
            self.history.append(entry)
            self.update_check_point()
        
        # Load best model at end
        if self.load_best_model_at_end and self.val_dataloader:
            metrics_history = [el[f"val_{self.best_metrics}"] for el in self.history]
            best_epoch = np.argmin(metrics_history) if self.best_metrics == "loss" else np.argmax(metrics_history)
            best_checkpoint = f"checkpoint-{best_epoch}"
            self.subgraph_generator.load_state_dict(
                torch.load(os.path.join())
            )
        return self.history
    
    def predict(self, test_dataloader=None):
        if not self.test_dataloader and test_dataloader is None:
            raise Exception("You should fill test_ds when initializing trainer if you want to predict or fill the test_dataloader params in this function")
        self.test_dataloader = self.test_dataloader or test_dataloader
        
        test_steps = math.ceil(len(self.test_dataloader.dataset) / self.batch_size)
        test_bar = tqdm(total=test_steps, desc="Test")
        test_metrics = self.run_epoch(self.test_dataloader, test_bar, train=False)

        for k in test_metrics.keys():
            test_metrics[k.replace("val", "test")] = test_metrics.pop(k)
        self.test_metrics = test_metrics
        return test_metrics
    
    def save(self, directory=None, save_history=True, save_evaluation_metrics=True):
        directory = directory or self.out_dir
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
        # Save model
        self.accelerator.wait_for_everyone()
        self.accelerator.save_model(self.subgraph_generator, os.path.join(directory, "model.pth"))
    
    def update_check_point(self):
        if len(self.history) > self.max_check_point:
            best_checkpoint = None
            if self.val_dataloader:
                metrics_history = [el[f"val_{self.best_metrics}"] for el in self.history]

                best_index = np.argmin(metrics_history) if self.best_metrics.endswith("loss") else np.argmax(metrics_history)
                best_checkpoint = f"checkpoint-{best_index}"
            # Delete Check point if exceed limit
            checkpoints = [el for el in os.listdir(self.out_dir) if re.match(r"checkpoint-\d+", el)
                        and os.path.isdir(os.path.join(self.out_dir, el)) and el != best_checkpoint]
            deleted_checkpoint = checkpoints.pop(0)
            shutil.rmtree(os.path.join(self.out_dir, deleted_checkpoint))
        
        # Save model
        current_epoch = len(self.history) - 1
        self.save(directory=os.path.join(self.out_dir, f"checkpoint-{current_epoch}"),
                  save_history=True,
                  save_evaluation_metrics=False)