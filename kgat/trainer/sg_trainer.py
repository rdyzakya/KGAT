from torch.nn import BCEWithLogitsLoss
from ..data import SubgraphGenerationCollator
import torch
from .utils import context_manager
from sklearn.metrics import classification_report
import time

from .trainer import Trainer

class SubgraphGenerationTrainer(Trainer):
    def __init__(self, 
                 pipeline,
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
                 best_metrics="sg_loss",
                 load_best_model_at_end=False,
                 optimizer="sgd",
                 optimizer_kwargs={},
                 neg_loss_weight=1.0):
        
        self.collate_fn = SubgraphGenerationCollator(tokenizer=tokenizer, 
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
                         alpha=alpha,
                         neg_loss_weight=neg_loss_weight)
        
    def criterion(self, preds, labels):
        # weight = torch.ones_like(labels)
        # neg_loss_weight = (labels == 1).sum() / (labels == 0).sum() if self.config.neg_loss_weight == "auto" else self.config.neg_loss_weight
        # weight[labels == 0] = neg_loss_weight

        crit = BCEWithLogitsLoss(weight=None)
        return crit(preds, labels)

    def create_score_matrix(self, n_entities, n_relations, x_coo, y_coo_cls=None):
        score_matrix = torch.zeros(n_entities, n_relations, n_entities, dtype=torch.float32, device=x_coo.device)
        x_coo = x_coo[y_coo_cls.bool()] if y_coo_cls is not None else x_coo
        score_matrix[x_coo[:,0], x_coo[:,1], x_coo[:,2]] = 1.0
        return score_matrix

    def compute_metrics(self, preds, labels, prefix=None):
        prefix = prefix or ""

        assert torch.logical_or(preds == 1, preds == 0).all(), f"The predictions value only allow 1 and 0, your prediction values are {preds.unique()}"
        assert torch.logical_or(labels == 1, labels == 0).all(), f"The label value only allow 1 and 0, your label values are {labels.unique()}"
        report = classification_report(y_pred=preds, y_true=labels)
        accuracy = report["accuracy"]
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1 = report["weighted avg"]["f1-score"]
        return {
            f"{prefix}accuracy" : accuracy,
            f"{prefix}precision" : precision,
            f"{prefix}recall" : recall,
            f"{prefix}f1" : f1,
        }
        # tp = torch.logical_and(preds == 1, labels == 1).sum()
        # tn = torch.logical_and(preds == 0, labels == 0).sum()
        # fp = torch.logical_and(preds == 1, labels == 0).sum()
        # fn = torch.logical_and(preds == 0, labels == 1).sum()

        # accuracy = (tp + tn) / (tp + tn + fp + fn)
        # precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
        # recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
        # f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)
        # return {
        #     f"{prefix}accuracy" : accuracy.item(),
        #     f"{prefix}precision" : precision.item(),
        #     f"{prefix}recall" : recall.item(),
        #     f"{prefix}f1" : f1.item(),
        # }

    def run_epoch(self, dataloader, bar, train=True):
        if train:
            self.pipeline.model.train()
            self.pipeline.lmkbc_model.freeze()
        else:
            self.pipeline.model.eval()
        self.pipeline.lmkbc_model.eval()

        sum_loss_sg = 0
        len_data_sg = 0

        sum_loss_gg = 0
        len_data_gg = 0

        all_sg_preds = []
        all_sg_labels = []

        all_gg_preds = []
        all_gg_labels = []

        start_time = time.time()

        for batch in dataloader:
            if train:
                self.optimizer.zero_grad()
            with torch.no_grad():
                queries = self.pipeline.lmkbc_model.batch_last_hidden_state(
                    input_ids=batch["graph_query_input_ids"],
                    attention_mask=batch["graph_query_attention_mask"],
                    batch_size=self.config.last_hidden_state_bsize
                )

                entities = self.pipeline.lmkbc_model.batch_last_hidden_state(
                    input_ids=batch["entities_input_ids"],
                    attention_mask=batch["entities_attention_mask"],
                    batch_size=self.config.last_hidden_state_bsize
                )

                relations = self.pipeline.lmkbc_model.batch_last_hidden_state(
                    input_ids=batch["relations_input_ids"],
                    attention_mask=batch["relations_attention_mask"],
                    batch_size=self.config.last_hidden_state_bsize
                )

            with context_manager(train=train):
                loss = 0
                x_coo = batch["x_coo"]

                if self.config.alpha > 0:
                    sg_out = self.pipeline.model(
                        queries=queries,
                        entities=entities,
                        relations=relations,
                        x_coo=x_coo,
                        batch=batch["batch"]
                    )

                    sg_labels = self.create_score_matrix(
                        n_entities=entities.shape[0],
                        n_relations=relations.shape[0],
                        x_coo=x_coo,
                        y_coo_cls=batch["y_coo_cls"]
                    )

                    filtered_sg_out = sg_out[x_coo[:,0], x_coo[:,1], x_coo[:,2]]
                    filtered_sg_labels = sg_labels[x_coo[:,0], x_coo[:,1], x_coo[:,2]]

                    sg_loss = self.criterion(filtered_sg_out, filtered_sg_labels)
                    
                    all_sg_preds.append(filtered_sg_out.sigmoid().round().int())
                    all_sg_labels.append(filtered_sg_labels.int())

                    loss += self.config.alpha * sg_loss
                    sum_loss_sg += sg_loss.item() * x_coo.shape[0]
                    len_data_sg += x_coo.shape[0]

                if self.config.alpha < 1.0:
                    gg_out = self.pipeline.model.encoder_decoder(
                        entities=entities,
                        relations=relations,
                        x_coo=x_coo
                    )

                    gg_labels = self.create_score_matrix(
                        n_entities=entities.shape[0],
                        n_relations=relations.shape[0],
                        x_coo=x_coo,
                        y_coo_cls=None
                    )

                    ### TODO filter berdasarkan batch, biar ga ada intersection antar batch
                    gg_loss = self.criterion(gg_out.view(-1), gg_labels.view(-1))

                    all_gg_preds.append(gg_out.view(-1).sigmoid().round().int())
                    all_gg_labels.append(gg_labels.view(-1).int())

                    loss += (1 - self.config.alpha) * gg_loss
                    sum_loss_gg += gg_loss.item() * (gg_out.shape[0] * gg_out.shape[1] * gg_out.shape[2])
                    len_data_gg += (gg_out.shape[0] * gg_out.shape[1] * gg_out.shape[2])
            if train:
                self.accelerator.backward(loss)
                self.optimizer.step()

            bar.update(1)
        
        end_time = time.time()


        prefix = "train_" if train else "val_"
        metrics = {
            f"{prefix}time" : end_time - start_time,
        }

        if self.config.alpha > 0:
            loss_sg = sum_loss_sg / len_data_sg
            all_sg_preds = torch.cat(all_sg_preds)
            all_sg_labels = torch.cat(all_sg_labels)
            metrics[f"{prefix}sg_loss"] = loss_sg
            metrics.update(
                self.compute_metrics(all_sg_preds, all_sg_labels, prefix=f"{prefix}sg_")
            )
        if self.config.alpha < 1.0:
            loss_gg = sum_loss_gg / len_data_gg
            all_gg_preds = torch.cat(all_gg_preds)
            all_gg_labels = torch.cat(all_gg_labels)
            metrics[f"{prefix}gg_loss"] = loss_gg
            metrics.update(
                self.compute_metrics(all_gg_preds, all_gg_labels, prefix=f"{prefix}gg_")
            )
        return metrics