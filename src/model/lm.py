from utils import KG_MASK
import torch

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py

class ModelForLMKBC(torch.nn.Module):
    def __init__(self, transformer, subgraphpooler, vt_transformer, clm_model, kg_id):
        super().__init__()
        self.transformer = transformer
        self.subgraphpooler = subgraphpooler
        self.vt_transformer = vt_transformer
        self.clm = clm_model
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
    
    def forward_clm(self, input_ids, attention_mask, virtual_tokens):
        all_inputs_embeds = []
        all_attention_masks = []

        for iid, am, vt in zip(input_ids, attention_mask, virtual_tokens):
            ie, am = self.prepare_one_input(iid, am, vt)
            all_inputs_embeds.append(ie)
            all_attention_masks.append(am)
        
        all_inputs_embeds = torch.stack(all_inputs_embeds)
        all_attention_masks = torch.stack(all_attention_masks)

        return self.clm.forward(inputs_embeds=all_inputs_embeds, attention_mask=attention_mask)
    
    def forward_entities_or_relations(self, input_ids, attention_mask):
        hidden_states = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = hidden_states[0]

        # get last token hidden state, as how any left-to-right model with seq-cls head do
        sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(hidden_states.device)

        batch_size = input_ids.shape[0]
        pooled_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

        return pooled_hidden_states
    
    def forward(self, text_input_ids, text_attention_mask,
                entities_input_ids, entities_attention_mask,
                relations_input_ids, relations_attention_mask,
                x_coo, batch):
        # text_in, entities, relations, x_coo, batch
        pass