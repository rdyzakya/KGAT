from torch.nn import CrossEntropyLoss, MSELoss
from ..data import LMKBCCollator
import torch
from .utils import context_manager
import time

from .trainer import Trainer

import math
from tqdm import tqdm

class LMKBCTrainer(Trainer):
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
                 beta1=1.0,
                 beta2=1.0):
        self.collate_fn = LMKBCCollator(tokenizer=tokenizer, 
                                        n_process=torch.cuda.device_count(),
                                        left=True)
        super().__init__(pipeline=pipeline,
                         tokenizer=tokenizer,
                         train_ds=train_ds,
                         val_ds=val_ds,
                         test_ds=test_ds,
                         epoch=epoch,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         last_hidden_state_bsize=last_hidden_state_bsize,
                         out_dir=out_dir,
                         max_check_point=max_check_point,
                         best_metrics=best_metrics,
                         load_best_model_at_end=load_best_model_at_end,
                         optimizer=optimizer,
                         optimizer_kwargs=optimizer_kwargs,
                         beta1=beta1,
                         beta2=beta2)
    
    def criterion(self, preds, labels, lmkbc=True):
        if lmkbc:
            crit = CrossEntropyLoss(ignore_index=-100)
            return crit(preds.view(-1, preds.shape[-1]), labels.view(-1))
        else:
            crit = MSELoss()
            return crit(preds, labels)
    
    def run_epoch(self, dataloader, bar, train=True):
        if train:
            self.pipeline.model.train()
            self.pipeline.lmkbc_model.freeze()
            self.pipeline.model.freeze_injector_and_encoder() 
            # we want to make the result explainable, so the module 
            # contained in subgraph generator will be in a freeze condition
        else:
            self.pipeline.model.eval()
        self.pipeline.lmkbc_model.eval()

        lmkbc_sum_loss = 0
        lmkbc_len_data = 0

        n_object_sum_loss = 0
        n_object_len_data = 0

        start_time = time.time()

        for batch in dataloader:
            if train:
                self.optimizer.zero_grad()
            with torch.no_grad():
                queries = self.pipeline.lmkbc_model.batch_last_hidden_state(
                    input_ids=batch["graph_query_input_ids"].to(self.lmkbc_model_device),
                    attention_mask=batch["graph_query_attention_mask"].to(self.lmkbc_model_device),
                    batch_size=self.config.last_hidden_state_bsize
                )

                entities = self.pipeline.lmkbc_model.batch_last_hidden_state(
                    input_ids=batch["entities_input_ids"].to(self.lmkbc_model_device),
                    attention_mask=batch["entities_attention_mask"].to(self.lmkbc_model_device),
                    batch_size=self.config.last_hidden_state_bsize
                )

                relations = self.pipeline.lmkbc_model.batch_last_hidden_state(
                    input_ids=batch["relations_input_ids"].to(self.lmkbc_model_device),
                    attention_mask=batch["relations_attention_mask"].to(self.lmkbc_model_device),
                    batch_size=self.config.last_hidden_state_bsize
                )
            
            queries = queries.to(self.model_device)
            entities = entities.to(self.model_device)
            relations = relations.to(self.model_device)

            with context_manager(train=train):
                vt_out, n_object_out = self.pipeline.model(
                        queries=queries,
                        entities=entities,
                        relations=relations,
                        x_coo=batch["x_coo"],
                        batch=batch["entities_batch"]
                    )
                
                logits = self.pipeline.lmkbc_model.forward_lmkbc(
                    batch["lmkbc_input_ids"], batch["lmkbc_attention_mask"], vt_out, batch=batch["graph_emb_batch"]
                )

                lmkbc_loss = self.criterion(logits.view(-1, logits.shape[-1]), batch["lmkbc_labels"], lmkbc=True)
                n_object_loss = self.criterion(n_object_out, batch["n_object"], lmkbc=False)

                loss = self.config.beta1 * lmkbc_loss + self.config.beta2 * n_object_loss
                # sum_loss += loss.item() * logits.shape[0]
                # len_data += logits.shape[0]
                lmkbc_sum_loss += lmkbc_loss.item() * logits.shape[0]
                lmkbc_len_data += logits.shape[0]

                n_object_sum_loss += n_object_loss.item() * n_object_loss.shape[0]
                n_object_len_data += n_object_loss.shape[0]
            if train:
                self.accelerator.backward(loss)
                self.optimizer.step()
            
            bar.update(1)
        
        end_time = time.time()

        prefix = "train_" if train else "val_"
        metrics = {
            f"{prefix}time" : end_time - start_time,
            f"{prefix}lmkbc_loss" : lmkbc_sum_loss / lmkbc_len_data,
            f"{prefix}n_object_loss" : n_object_loss / n_object_len_data
        }

        return metrics
    
    def predict(self, test_dataloader=None):
        if not self.test_dataloader and test_dataloader is None:
            raise Exception("You should fill test_ds when initializing trainer if you want to predict or fill the test_dataloader params in this function")
        self.test_dataloader = self.test_dataloader or test_dataloader
        
        test_steps = math.ceil(len(self.test_dataloader.dataset) / self.config.batch_size)
        test_bar = tqdm(total=test_steps, desc="Test")
        # test_metrics = self.run_epoch(self.test_dataloader, test_bar, train=False)
        for batch in self.test_dataloader:
            with torch.no_grad():
                queries = self.pipeline.lmkbc_model.batch_last_hidden_state(
                    input_ids=batch["graph_query_input_ids"].to(self.lmkbc_model_device),
                    attention_mask=batch["graph_query_attention_mask"].to(self.lmkbc_model_device),
                    batch_size=self.config.last_hidden_state_bsize
                )

                entities = self.pipeline.lmkbc_model.batch_last_hidden_state(
                    input_ids=batch["entities_input_ids"].to(self.lmkbc_model_device),
                    attention_mask=batch["entities_attention_mask"].to(self.lmkbc_model_device),
                    batch_size=self.config.last_hidden_state_bsize
                )

                relations = self.pipeline.lmkbc_model.batch_last_hidden_state(
                    input_ids=batch["relations_input_ids"].to(self.lmkbc_model_device),
                    attention_mask=batch["relations_attention_mask"].to(self.lmkbc_model_device),
                    batch_size=self.config.last_hidden_state_bsize
                )
            
                queries = queries.to(self.model_device)
                entities = entities.to(self.model_device)
                relations = relations.to(self.model_device)

                vt_out, n_object_out = self.pipeline.model(
                        queries=queries,
                        entities=entities,
                        relations=relations,
                        x_coo=batch["x_coo"],
                        batch=batch["entities_batch"]
                    )

                # generate_lmkbc(self, input_ids, attention_mask, graph_embeddings, batch=None, **kwargs)
                generation_result = self.pipeline.lmkbc_model.generate_lmkbc(input_ids, 
                                                                             attention_mask, 
                                                                             graph_embeddings, 
                                                                             batch=None,
                                                                             num_beams=,
                                                                             num_return_sequences=4,
                                                                             )
                
                # num_beams = len(objects)
                # do sample? No i guess
                # return sequence = len(objects)
                
                # logits = self.pipeline.lmkbc_model.forward_lmkbc(
                #     batch["lmkbc_input_ids"], batch["lmkbc_attention_mask"], vt_out, batch=batch["graph_emb_batch"]
                # )

                # loss = self.criterion(logits.view(-1, logits.shape[-1]), batch["lmkbc_labels"])
                # sum_loss += loss.item() * logits.shape[0]
                # len_data += logits.shape[0]
            test_bar.update(1)

        # RuntimeError: dictionary keys changed during iteration
        new_metrics = {}
        for k in test_metrics.keys():
            new_metrics[k.replace("val", "test")] = test_metrics[k]
        self.test_metrics = new_metrics
        return self.test_metrics