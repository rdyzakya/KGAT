import torch
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.utils.data import DataLoader
from kgat.data import SubgraphGenerationCollator
from torch.nn import BCEWithLogitsLoss

class SGTrainer:
    def __init__(self, model,
                 tokenizer,
                 train_ds,
                 scheduler=None, 
                 val_ds=None,
                 left=True,
                 train_batch_size=8,
                 val_batch_size=8,
                 epoch=10):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self.model = model
        self.train_ds = train_ds
        self.criterion = BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        self.device = device
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
        preds = preds.sigmoid().round() # sigmoid -> round

        metrics = classification_report(y_true=labels, y_pred=preds, output_dict=True)

        return {
            "eval_accuracy" : metrics["accuracy"],
            "eval_recall" : metrics["weighted avg"]["recall"],
            "eval_precision" : metrics["weighted avg"]["precision"],
            "eval_f1_score" : metrics["weighted avg"]["f1-score"]
        }

    def run_epoch(self, dataloader):
        self.model.train()
        loss_data = torch.zeros(2).to(self.device)
        for batch in dataloader:
            labels = batch.pop("y_coo_cls")
            labels = labels.to(self.device)

            for k, v in batch:
                batch[k] = v.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            loss_data[0] += loss
            loss_data[1] += len(labels)
        return loss_data[0] / loss_data[1]

    def evaluation_loop(self, dataloader):
        self.model.eval()
        loss_data = torch.zeros(2).to(self.device)
        y_pred = []
        y_true = []
        
        for batch in tqdm(dataloader, desc="Evaluation"):
            labels = batch.pop("y_coo_cls")
            labels = labels.to(self.device)

            for k, v in batch:
                batch[k] = v.to(self.device)

            outputs = self.model(**batch)
            loss = self.criterion(outputs, labels)

            loss_data[0] += loss
            loss_data[1] += len(labels)

            y_pred.append(outputs)
            y_true.append(labels)
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)

        metrics = self.compute_metrics(y_pred, y_true)

        metrics.update({
            "eval_loss" : loss_data[0] / loss_data[1]
        })

        return metrics

    def train_loop(self):
        train_dataloader = self.prepare_dataloader(self.train_ds, self.train_batch_size)
        val_dataloader = self.prepare_dataloader(self.val_ds, self.val_batch_size) if self.val_ds else None
        
        history = []

        for e in tqdm(range(self.epoch), desc="Training"):
            train_loss = self.run_epoch(train_dataloader)
            data = {
                "epoch" : e+1,
                "train_loss" : train_loss
            }
            
            if val_dataloader:
                eval_metrics = self.evaluation_loop(val_dataloader)
                data.update(eval_metrics)

            print(data)
            history.append(data)

        return self.model, history