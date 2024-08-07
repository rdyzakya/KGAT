from abc import ABC
from ...utils import Mask
import torch
from transformers import EosTokenCriteria, StoppingCriteriaList

class LMKBCWrapper(ABC):
    @property
    def backbone(self):
        raise NotImplementedError("This is an abstract class")
    
    @property
    def embeddings(self):
        raise NotImplementedError("This is an abstract class")
    
    @property
    def embed_dim(self):
        return self.embeddings.weight.shape[-1]
    
    def last_hidden_state(self, input_ids, attention_mask):

        hidden_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = hidden_states[0]

        # get last token hidden state, as how any left-to-right model with seq-cls head do
        sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(hidden_states.device)

        batch_size = input_ids.shape[0]
        pooled_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

        return pooled_hidden_states
    
    def batch_last_hidden_state(self, input_ids, attention_mask, batch_size=16):
        res = []

        for i in range(0, input_ids.shape[0], batch_size):
            current_inputs_ids = input_ids[i:i+batch_size]
            current_attention_mask = attention_mask[i:i+batch_size]

            current_hs = self.last_hidden_state(current_inputs_ids, current_attention_mask)

            res.append(current_hs)
        
        res = torch.vstack(res)

        return res
    
    def prepare_tokenizer(self, tokenizer):
        # Add padding
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.config.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "left"

        # Add KG special token
        self.config.kg_token = Mask.KG_MASK
        tokenizer.kg_token = self.config.kg_token
        tokenizer.add_special_tokens({
            "additional_special_tokens" : [self.config.kg_token]
        })

        vocab = tokenizer.get_vocab()
        kg_token_id = vocab[self.config.kg_token]
        self.config.kg_token_id = kg_token_id
        tokenizer.kg_token_id = kg_token_id

        self.config.eos_token = tokenizer.eos_token
        self.config.eos_token_id = tokenizer.eos_token_id

        return tokenizer
    
    def prepare_lmkbc(self, input_ids, attention_mask, graph_embeddings):
        mask = input_ids == self.config.kg_token_id

        input_ids[mask] = 0 # change to 0, because we don't resize the params

        embeds = self.embeddings(input_ids)
        embeds[mask] = graph_embeddings.view(-1, graph_embeddings.shape[-1])

        return embeds, attention_mask
    
    def forward_lmkbc(self, input_ids, attention_mask, graph_embeddings, batch=None):
        batch = torch.arange(0,input_ids.shape[0]) if batch is None else batch
        graph_embeddings = graph_embeddings[batch]
        result_embeds, result_attention_mask = self.prepare_lmkbc(input_ids, attention_mask, graph_embeddings)
        return self.forward(inputs_embeds=result_embeds, attention_mask=result_attention_mask)
    
    def generate_lmkbc(self, input_ids, attention_mask, graph_embeddings, batch=None, **kwargs):
        batch = torch.arange(0,input_ids.shape[0]) if batch is None else batch
        graph_embeddings = graph_embeddings[batch]
        result_embeds, result_attention_mask = self.prepare_lmkbc(input_ids, attention_mask, graph_embeddings)

        return self.generate(inputs_embeds=result_embeds, attention_mask=result_attention_mask, **kwargs)
    
    def freeze(self):
        # Freeze
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.embeddings.parameters():
            param.requires_grad = False