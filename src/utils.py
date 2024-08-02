KG_MASK = "<KNOWLEDGE_GRAPH>"
SUBJECT_MASK = "<SUBJECT>"
RELATION_MASK = "<RELATION>"
OBJECT_MASK = "<OBJECT>"
EOS_MASK = "<EOS>"

EMPTY_OBJECT = "NONE"

VALID_MASK = "<VALID>"
TRUE_FLAG = "TRUE"
FALSE_FLAG = "FALSE"

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, greater_is_better=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.greater_is_better = greater_is_better
        self.pivot_val_metrics = float('inf') if not greater_is_better else -float('inf')

    def early_stop(self, val_metrics):
        if not self.greater_is_better:
            if val_metrics < self.pivot_val_metrics:
                self.pivot_val_metrics = val_metrics
                self.counter = 0
            elif val_metrics > (self.pivot_val_metrics + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        else:
            if val_metrics > self.pivot_val_metrics:
                self.pivot_val_metrics = val_metrics
                self.counter = 0
            elif val_metrics < (self.pivot_val_metrics - self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        return False

import os
import shutil
import torch
import json

class SaveAndLoad:
    def __init__(self, model, out_dir, metrics_name, max_ckpt=None, greater_is_better=False):
        self.model = model
        self.max_ckpt = max_ckpt
        self.greater_is_better = greater_is_better
        self.out_dir = out_dir
        self.metrics_name = metrics_name
    
    def load_best(self, history):
        best_epoch = -1
        best_metrics = float('inf') if not self.greater_is_better else -float('inf')
        for i, el in enumerate(history):
            if (el[self.metrics_name] > best_metrics and self.greater_is_better) or (el[self.metrics_name] < best_metrics and not self.greater_is_better):
                best_epoch = i
        ckpt_dir = os.path.join(self.out_dir, f"ckpt-{best_epoch+1}", "model.pth")
        return self.model.load_state_dict(
            torch.load(ckpt_dir)["state_dict"]
        )

    def save(self, history, is_ckpt=True):
        # SAVE
        ckpt_dir = os.path.join(self.out_dir, f"ckpt-{len(history)}") if is_ckpt else self.out_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, "history.json"), 'w') as fp:
            json.dump(history, fp)
        torch.save({
            "attribute" : self.model.save_attribute(),
            "state_dict" : self.model.state_dict()
        }, os.path.join(ckpt_dir, "model.pth"))
        
        # DELETE
        if self.max_ckpt and is_ckpt:
            ckpt_folders = [el for el in os.listdir(self.out_dir) if el.startswith("ckpt-") and os.path.isdir(el)]
            if len(ckpt_folders) > self.max_ckpt:
                epochs = list(range(len(history)))
                metrics = [el[self.metrics_name] for el in history]

                epochs_and_metrics = sorted(zip(epochs, metrics), reverse=not self.greater_is_better, key=lambda x: x[1])
                
                worst_eam = epochs_and_metrics[:len(history) - self.max_ckpt]

                for epoch, _ in worst_eam:
                    folder_path = os.path.join(self.out_dir, f"ckpt-{epoch+1}")
                    if os.path.exists(folder_path):
                        shutil.rmtree(folder_path)