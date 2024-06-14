from transformers import Trainer
# from trl import SFTTrainer
from torch.nn import (
    BCELoss,
    BCEWithLogitsLoss
)

class SGTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = BCEWithLogitsLoss(reduction="mean")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # graph_query = tokenizer(batch[0], padding=True, truncation=True, max_length=64, return_tensors="pt")
        # entities = tokenizer(batch[1], padding=True, truncation=True, max_length=16, return_tensors="pt")
        # relations = tokenizer(batch[2], padding=True, truncation=True, max_length=16, return_tensors="pt")
        # x_coo = batch[3]
        # node_batch = batch[4]
        # y_coo = batch[5]

        # text, entities, relations, x_coo, y_coo_cls
        label = inputs.pop("y_coo_cls")
        mean_fused_score, _, _ = model(**inputs)

        assert len(mean_fused_score) == len(label)

        mask = label != -1

        mean_fused_score = mean_fused_score[mask]
        label = label[mask]

        label = label.float()

        loss = self.criterion(mean_fused_score, label)

        if return_outputs:
            return loss, mean_fused_score
        return loss