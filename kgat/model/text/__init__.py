from .vt_transform import (
    SubgraphVTTransformation
)

import torch

class TextModule(torch.nn.Module):
    def __init__(self, vt_transformer, clm_embedding, clm_model, kg_id):
        super().__init__()
        self.vt_transformer = vt_transformer
        self.clm_embedding = clm_embedding
        self.clm = clm_model
        self.kg_id = kg_id
    
    def prepare_one_input(self, input_id, attention_mask, virtual_token):
        kg_idx = torch.argwhere(input_id == self.kg_id).flatten()
        prev_k = 0
        input_embeds = []
        for k in kg_idx:
            input_embeds.append(self.clm_embedding(input_id[prev_k:k]))
            input_embeds.append(virtual_token)
            prev_k = k + 1
        input_embeds.append(self.clm_embedding(input_id[prev_k:]))
        n_added_vects = (virtual_token.shape[0]-1) * kg_idx.shape[0]

        input_embeds = torch.vstack(input_embeds)[n_added_vects:] # padding
        attention_mask = torch.cat([attention_mask, torch.ones(n_added_vects, dtype=attention_mask.dtype)])[n_added_vects:]

        return input_embeds, attention_mask
    
    def forward_clm(self, input_ids, attention_mask, virtual_tokens):
        all_inputs_embeds = []
        all_attention_masks = []

        for iid, am, vt in zip(input_ids, attention_mask, virtual_tokens):
            ie, am = self.prepare_one_input(iid, am, vt)
            all_inputs_embeds.append(ie)
            all_attention_masks.append(am)
        
        all_inputs_embeds = torch.stack(all_inputs_embeds)
        all_attention_masks = torch.stack(all_attention_masks)

        model_inputs = self.clm.prepare_inputs_for_generation(
            input_ids=input_ids,
            inputs_embeds=all_inputs_embeds,
            attention_mask=all_attention_masks
        )

        return self.clm.forward(**model_inputs)
    
    def forward(self, prompt_input_ids, prompt_attention_mask, subgraph_emb):
        # TRANSFORM TO VT
        vt = self.vt_transformer(subgraph_emb)
        # INSERT TO CLM
        return self.forward_clm(prompt_input_ids, prompt_attention_mask, vt)
    
    def freeze_llm(self):
        for param in self.clm_embedding.parameters():
            param.requires_grad = False
        for param in self.clm.parameters():
            param.requires_grad = False