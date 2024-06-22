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
        
    def criterion(self, preds, labels, nlw=1.0):
        weight = torch.ones_like(labels)
        neg_loss_weight = (labels == 1).sum() / (labels == 0).sum() if nlw == "auto" else nlw
        weight[labels == 0] = neg_loss_weight

        crit = BCEWithLogitsLoss(weight=weight)
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
        # report = classification_report(y_pred=preds.tolist(), y_true=labels.tolist(), output_dict=True)
        # accuracy = report["accuracy"]
        # precision = report["macro avg"]["precision"]
        # recall = report["macro avg"]["recall"]
        # f1 = report["macro avg"]["f1-score"]

        tp = torch.logical_and(labels == 1, preds == 1).sum()
        tn = torch.logical_and(labels == 0, preds == 0).sum()
        fp = torch.logical_and(labels == 0, preds == 1).sum()
        fn = torch.logical_and(labels == 1, preds == 0).sum()

        accuracy = ((tp + tn) / (tp + tn + fp + fn)).item() if (tp + tn + fp + fn) > 0 else 0.0
        
        precision_1 = ((tp) / (tp + fp)).item() if (tp + fp) > 0 else 0.0
        recall_1 = ((tp) / (tp + fn)).item() if (tp + fn) > 0 else 0.0
        f1_1 = ((2 * precision_1 * recall_1) / (precision_1 + recall_1)) if (precision_1 + recall_1) > 0 else 0.0

        precision_0 = ((tn) / (tn + fn)).item() if (tn + fn) > 0 else 0.0
        recall_0 = ((tn) / (tn + fp)).item() if (tn + fp) > 0 else 0.0
        f1_0 = ((2 * precision_0 * recall_0) / (precision_0 + recall_0)) if (precision_0 + recall_0) > 0 else 0.0
        
        # macro avg
        # https://docs.kolena.com/metrics/averaging-methods/#:~:text=If%20you%20want%20to%20treat,average%20instead%20of%20macro%20average.
        # If you want to treat all classes equally, then using macro average would be a good choice. 
        # If you have an imbalanced dataset but want to assign more weight to classes with more 
        # samples, consider using weighted average instead of macro average.
        precision = (precision_0 + precision_1) / 2
        recall = (recall_0 + recall_1) / 2
        f1 = (f1_0 + f1_1) / 2
        return {
            f"{prefix}accuracy" : accuracy,
            f"{prefix}precision" : precision,
            f"{prefix}recall" : recall,
            f"{prefix}f1" : f1,
        }

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

                    sg_loss = self.criterion(filtered_sg_out, filtered_sg_labels, nlw=1.0)
                    
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

                    gg_out = gg_out.transpose(0,1)[:,batch["batch"].unsqueeze(-1) == batch["batch"]].view(-1)
                    gg_labels = gg_labels.transpose(0,1)[:,batch["batch"].unsqueeze(-1) == batch["batch"]].view(-1)
                    
                    gg_loss = self.criterion(gg_out, gg_labels, nlw=self.config.neg_loss_weight)

                    all_gg_preds.append(gg_out.sigmoid().round().int())
                    all_gg_labels.append(gg_labels.int())

                    loss += (1 - self.config.alpha) * gg_loss
                    sum_loss_gg += gg_loss.item() * gg_out.shape[0]
                    len_data_gg += gg_out.shape[0]
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