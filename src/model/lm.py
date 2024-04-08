from transformers import MistralForCausalLM, MistralPreTrainedModel
from utils import KG_MASK
import torch

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py

class MistralForLMKBC(MistralPreTrainedModel):
    def __init__(self, config, kg_id):
        super().__init__(config)
        self.clm = MistralForCausalLM(config)
        self.kg_id = kg_id
    
    def prepare_one_input(self, input_id, attention_mask, virtual_token):
        kg_idx = torch.argwhere(input_id == self.kg_id).flatten()
        prev_k = 0
        input_embeds = []
        for k in kg_idx:
            input_embeds.append(self.model.embed_tokens(input_id[prev_k:k]))
            input_embeds.append(virtual_token)
            prev_k = k + 1
        n_added_vects = virtual_token.shape[0] * kg_idx.shape[0]

        input_embeds = torch.vstack(input_embeds)[n_added_vects:] # padding
        attention_mask = torch.cat([attention_mask, torch.ones(n_added_vects, dtype=attention_mask.dtype)])[n_added_vects:]

        return input_embeds, attention_mask
    
    def forward(self, input_ids, attention_mask, virtual_tokens):
        all_inputs_embeds = []
        all_attention_masks = []

        for iid, am, vt in zip(input_ids, attention_mask, virtual_tokens):
            ie, am = self.prepare_one_input(iid, am, vt)
            all_inputs_embeds.append(ie)
            all_attention_masks.append(am)
        
        all_inputs_embeds = torch.stack(all_inputs_embeds)
        all_attention_masks = torch.stack(all_attention_masks)

        return self.clm.forward(inputs_embeds=all_inputs_embeds, attention_mask=attention_mask)