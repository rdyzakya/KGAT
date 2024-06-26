import torch
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.utils.data import DataLoader
from kgat.data import SubgraphGenerationCollator
from torch.nn import BCEWithLogitsLoss
import math

from accelerate import Accelerator

class SGTrainer:
    def __init__(self, model,
                 tokenizer,
                 train_ds,
                 scheduler=None, 
                 val_ds=None,
                 left=True,
                 train_batch_size=8,
                 val_batch_size=8,
                 epoch=10,
                 gradient_accumulation_steps=1):
        
        self.accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model.to(device)

        self.model = model
        self.train_ds = train_ds
        self.criterion = BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        self.device = self.accelerator.device
        self.scheduler = scheduler
        self.val_ds = val_ds

        self.collate_fn = SubgraphGenerationCollator(tokenizer=tokenizer,
                                                     n_process=max(1, torch.cuda.device_count()),
                                                     left=left)
        
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.epoch = epoch
    
    def prepare_dataloader(self, ds, batch_size):
        return DataLoader(dataset=ds, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=self.collate_fn)

    def compute_metrics(self, preds, labels):
        # preds = preds.sigmoid().round() # sigmoid -> round

        metrics = classification_report(y_true=labels, y_pred=preds, output_dict=True)

        return {
            "eval_accuracy" : metrics["accuracy"],
            "eval_recall" : metrics["weighted avg"]["recall"],
            "eval_precision" : metrics["weighted avg"]["precision"],
            "eval_f1_score" : metrics["weighted avg"]["f1-score"]
        }

    def run_epoch(self, dataloader, pbar):
        self.model.train()
        loss_data = torch.zeros(2).to(self.device)
        for batch in dataloader:
            labels = batch.pop("y_coo_cls")
            # labels = labels.to(self.device)

            # for k, v in batch.items():
            #     batch[k] = v.to(self.device)
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                outputs, _, _ = self.model(**batch)
                loss = self.criterion(outputs, labels.float())
                # loss.backward()
                self.accelerator.backward(loss)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
            loss_data[0] += loss
            loss_data[1] += len(labels)

            pbar.update(1)
        return loss_data[0] / loss_data[1]

    def evaluation_loop(self, dataloader):
        self.model.eval()
        loss_data = torch.zeros(2).to(self.device)
        y_pred = []
        y_true = []
        
        for batch in tqdm(dataloader, desc="Evaluation"):
            labels = batch.pop("y_coo_cls")
            # labels = labels.to(self.device)

            # for k, v in batch.items():
            #     batch[k] = v.to(self.device)

            with torch.no_grad():
                outputs, _, _ = self.model(**batch)
            loss = self.criterion(outputs, labels.float())

            loss_data[0] += loss
            loss_data[1] += len(labels)

            y_pred.extend(outputs.sigmoid().round().long().tolist())
            y_true.extend(labels.tolist())

        metrics = self.compute_metrics(y_pred, y_true)

        metrics.update({
            "eval_loss" : (loss_data[0] / loss_data[1]).item()
        })

        return metrics

    def train_loop(self):

        train_steps = math.ceil(len(self.train_ds) / self.train_batch_size) * self.epoch
        pbar = tqdm(total=train_steps, desc="Training")

        train_dataloader = self.prepare_dataloader(self.train_ds, self.train_batch_size)
        val_dataloader = self.prepare_dataloader(self.val_ds, self.val_batch_size) if self.val_ds else None

        self.model, self.optimizer, self.scheduler, train_dataloader, val_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, train_dataloader, val_dataloader
        )
        
        history = []

        for e in range(self.epoch):
            train_loss = self.run_epoch(train_dataloader, pbar)
            data = {
                "epoch" : e+1,
                "train_loss" : train_loss.item()
            }
            
            if val_dataloader:
                eval_metrics = self.evaluation_loop(val_dataloader)
                data.update(eval_metrics)

            print(data)
            history.append(data)

        return self.model, history
    
    def save_model(self, model_config, out_path):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process():
            self.model = self.accelerator.unwrap_model(self.model)
            result = {
                "structure" : model_config["structure"],
                "state_dict" : {
                    "graph_module" : {}
                }
            }

            result["state_dict"]["graph_module"]["graphpooler"] = self.model.graphpooler.state_dict()
            result["state_dict"]["graph_module"]["subgraphpooler"] = self.model.subgraphpooler.state_dict()

            torch.save(result, out_path)