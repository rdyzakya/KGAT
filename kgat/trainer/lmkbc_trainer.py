# text_in, graph_query, entities, relations, x_coo, text_out

from transformers import Trainer, Seq2SeqTrainer
from torch.nn import (
    CrossEntropyLoss
)

