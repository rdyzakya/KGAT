from torch.nn import CrossEntropyLoss, MSELoss
from ..data import LMKBCCollator
import torch
from .utils import context_manager
from ..utils import NULL_SYM, post_process
import time

from .trainer import Trainer

import math
from tqdm import tqdm

import re

pattern = r"(.+)\|\s*(true|false)"

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
                 logging_steps=None,
                 beam_size=6,
                 max_length=32,
                 beta1=1.0,
                 beta2=1.0):
        self.collate_fn = LMKBCCollator(tokenizer=tokenizer, 
                                        n_process=torch.cuda.device_count(),
                                        left=True)
        self.test_collate_fn = LMKBCCollator(tokenizer=tokenizer, 
                                        n_process=torch.cuda.device_count(),
                                        left=True,
                                        test=True)
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
                         beta2=beta2,
                         logging_steps=logging_steps,
                         beam_size=beam_size,
                         max_length=max_length,)
    
    def criterion(self, preds, labels, lmkbc=True):
        if lmkbc:
            crit = CrossEntropyLoss(ignore_index=-100, reduction="none")
            return crit(preds.view(-1, preds.shape[-1]), labels.view(-1))
        else:
            # /raid/m13519061/anaconda3/envs/absa/lib/python3.10/site-packages/torch/nn/modules/loss.py:536: 
            # UserWarning: Using a target size (torch.Size([2])) that is different to the input size (torch.Size([2, 1])). 
            # This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
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
                vt_out = self.pipeline.model(
                        queries=queries,
                        entities=entities,
                        relations=relations,
                        x_coo=batch["x_coo"],
                        batch=batch["entities_batch"]
                    )
                
                logits = self.pipeline.lmkbc_model.forward_lmkbc(
                    batch["lmkbc_input_ids"], batch["lmkbc_attention_mask"], vt_out, batch=batch["graph_emb_batch"]
                )[0]

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["lmkbc_labels"][..., 1:].contiguous()
                shift_weights = batch["weights"][..., 1:].contiguous()

                lmkbc_loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), lmkbc=True)
                lmkbc_loss = lmkbc_loss * shift_weights.view(-1)
                lmkbc_loss = lmkbc_loss[shift_labels.view(-1) != -100].mean()

                loss = lmkbc_loss

                lmkbc_sum_loss += lmkbc_loss.item() * logits.shape[0]
                lmkbc_len_data += logits.shape[0]

            if train:
                self.accelerator.backward(loss)
                self.optimizer.step()

                if self.logging_steps:
                    self.steps += 1
                    if self.steps % self.logging_steps == 0:
                        log_message = []
                        loss_sg = lmkbc_sum_loss / lmkbc_len_data
                        log_message.append(f"loss_lmkbc : {loss_sg}")
                        self.log(" | ".join(log_message))
            
            bar.update(1)
        
        end_time = time.time()

        prefix = "train_" if train else "val_"
        metrics = {
            f"{prefix}time" : end_time - start_time,
            f"{prefix}lmkbc_loss" : lmkbc_sum_loss / lmkbc_len_data,
        }

        return metrics, None
    
    def compute_metrics(self, preds, labels):
        tp = 0
        fp = 0
        fn = 0

        preds = [[post_process(el2[0].lower()) for el2 in el1 if el2[1] == "true"] for el1 in preds]
        labels = [[el2.lower() for el2 in el1] for el1 in labels]

        preds = [list(set(el)) for el in preds]
        labels = [list(set(el)) for el in labels]

        for pred, label in zip(preds, labels):
            for p in pred:
                if p in label:
                    tp += 1
                else:
                    fp += 1
            if len(label) == 0 and label[0] == NULL_SYM:
                continue
            for l in label:
                if l not in pred:
                    fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "precision" : precision,
            "recall" : recall,
            "f1" : f1
        }
    
    def predict(self, test_dataloader=None):
        super().predict()
        if not self.test_dataloader and test_dataloader is None:
            raise Exception("You should fill test_ds when initializing trainer if you want to predict or fill the test_dataloader params in this function")
        self.test_dataloader = self.test_dataloader or test_dataloader
        
        test_steps = math.ceil(len(self.test_dataloader.dataset) / self.config.batch_size)
        test_bar = tqdm(total=test_steps, desc="Test")
        all_lmkbc_preds = []
        all_lmkbc_labels = []
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

                vt_out = self.pipeline.model(
                        queries=queries,
                        entities=entities,
                        relations=relations,
                        x_coo=batch["x_coo"],
                        batch=batch["entities_batch"]
                    )

                generation_result = self.pipeline.lmkbc_model.generate_lmkbc(
                                        batch["lmkbc_input_ids"],
                                        batch["lmkbc_attention_mask"],
                                        vt_out,
                                        batch=batch["graph_emb_batch"],
                                        num_beams=self.config.beam_size,
                                        num_return_sequences=self.config.beam_size,
                                        max_length=batch["lmkbc_input_ids"].shape[-1] + self.config.max_length
                                    )
                
                generation_result = self.tokenizer.batch_decode(generation_result, skip_special_tokens=True)

                for i in range(len(generation_result)):
                    m = re.match(pattern, generation_result[i])
                    if m is None:
                        generation_result[i] = (generation_result[i], "false")
                    else:
                        generation_result[i] = (m.group(1).strip(), m.group(2).strip())
                
                generation_result = [generation_result[i:i+self.config.beam_size] for i in range(0,len(generation_result),self.config.beam_size)]
                
            all_lmkbc_preds.extend(generation_result)
            all_lmkbc_labels.extend(batch["objects"])
            test_bar.update(1)

        self.test_metrics = self.compute_metrics(all_lmkbc_preds, all_lmkbc_labels)
        self.prediction_result = {
            "lmkbc" : self.construct_preds_labels(all_lmkbc_preds, all_lmkbc_labels)
        }
        return self.test_metrics, self.prediction_result
    
    def architecture(self, unwrapped_model):
        return dict(
                # input_dim = unwrapped_model.injector.input_dim,
                # encoder_h_dim = unwrapped_model.encoder.h_dim,
                # out_dim = unwrapped_model.encoder.h_dim,
                n_features=unwrapped_model.n_features,
                h_dim=unwrapped_model.h_dim,
                n_encoder_head = unwrapped_model.encoder.n_head,
                n_injector_head = unwrapped_model.injector.n_head,
                injector_dropout_p = unwrapped_model.injector.p,
                encoder_dropout_p = unwrapped_model.encoder.p,
                n_encoder_layers = unwrapped_model.encoder.n_layers,
                n_virtual_token = unwrapped_model.virtual_token.n_virtual_token,
                gnn_type=unwrapped_model.gnn_type,
                mp=unwrapped_model.mp,
                inject_edge_attr=unwrapped_model.inject_edge_attr
            )