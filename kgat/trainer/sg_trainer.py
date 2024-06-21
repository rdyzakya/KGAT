from torch.nn import BCEWithLogitsLoss
from ..data import SubgraphGenerationCollator
from sklearn.metrics import classification_report
import torch
import utils
import time

from .trainer import Trainer

class SubgraphGenerationTrainer(Trainer):
    def __init__(self, 
                 model, 
                 lmkbc_model, 
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
                 load_best_model_at_end=False,
                 optimizer="sgd",
                 optimizer_kwargs={}):
        
        self.collate_fn = SubgraphGenerationCollator(tokenizer=tokenizer, 
                                                     n_process=torch.cuda.device_count(), 
                                                     left=True)
        self.criterion = BCEWithLogitsLoss()
        super().__init__(model=model,
                         lmkbc_model=lmkbc_model,
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
                         alpha=alpha)

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
            self.model.train()
        else:
            self.model.eval()
        self.lmkbc_model.eval()

        loss_data = torch.zeros(2, dtype=torch.float32)

        all_sg_preds = []
        all_sg_labels = []

        all_gg_preds = []
        all_gg_labels = []

        start_time = time.time()

        for batch in dataloader:
            if train:
                self.optimizer.zero_grad()
            with torch.no_grad():
                queries = self.lmkbc_model.batch_last_hidden_state(
                    input_ids=batch["graph_query_input_ids"],
                    attention_mask=batch["graph_query_attention_mask"],
                    batch_size=self.config.last_hidden_state_bsize
                )

                entities = self.lmkbc_model.batch_last_hidden_state(
                    input_ids=batch["entities_input_ids"],
                    attention_mask=batch["entities_attention_mask"],
                    batch_size=self.config.last_hidden_state_bsize
                )

                relations = self.lmkbc_model.batch_last_hidden_state(
                    input_ids=batch["relations_input_ids"],
                    attention_mask=batch["relations_attention_mask"],
                    batch_size=self.config.last_hidden_state_bsize
                )
            with utils.context_manager(train=train):
                sg_out = self.model(
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

                loss = self.config.alpha * sg_loss

                if self.config.alpha < 1.0:
                    gg_out = self.model.encoder_decoder(
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

                    loss += (1 - self.config.alpha) * gg_loss
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
        if self.config.alpha < 1.0:
            metrics.update(
                self.compute_metrics(all_gg_preds, all_gg_labels, prefix=f"{prefix}gg_")
            )
        return metrics