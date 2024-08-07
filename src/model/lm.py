from abc import ABC
from transformers import (
    GPT2LMHeadModel,
    LlamaForCausalLM,
    MistralForCausalLM,
    OPTForCausalLM,
    AutoConfig,
)
import torch
from utils import KG_MASK

class LanguageModelForLMKBC(ABC):
    @property
    def backbone(self):
        raise NotImplementedError("This is an abstract class")
    
    @property
    def embeddings(self):
        raise NotImplementedError("This is an abstract class")
    
    @property
    def embed_dim(self):
        return self.embeddings.weight.shape[-1]
    
    def freeze(self):
        # Freeze
        for param in self.parameters():
            param.requires_grad = False
    
    def text_embedding(self, input_ids, attention_mask, index=None, n_tokens=1):
        # get last token hidden state, as how any left-to-right model with seq-cls head do
        sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        
        batch_size = input_ids.shape[0]

        inputs_embeds = self.embeddings(input_ids)

        current_inputs_embeds = inputs_embeds.clone()
        current_attention_mask = attention_mask.clone()
        for i in range(n_tokens):
            out = self.backbone(inputs_embeds=current_inputs_embeds, attention_mask=current_attention_mask, output_hidden_states=True)
            last_hidden_state = out[0]

            added_inputs_embeds = []

            for j in range(i+1):
                seq_len = sequence_lengths + j
                last_emb = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), seq_len.to(last_hidden_state.device)]
                added_inputs_embeds.append(last_emb)
            
            added_inputs_embeds = torch.stack(added_inputs_embeds)
            added_inputs_embeds = added_inputs_embeds.transpose(0,1)
            added_inputs_embeds = added_inputs_embeds.to(inputs_embeds.device)

            current_inputs_embeds = torch.hstack([inputs_embeds, added_inputs_embeds])
            current_attention_mask = torch.hstack([attention_mask, torch.ones(batch_size, i+1, device=attention_mask.device)])

            assert current_inputs_embeds.shape[1] == inputs_embeds.shape[1] + (i+1)
            assert current_attention_mask.shape[1] == attention_mask.shape[1] + (i+1)

        pooled_hidden_states = tuple(
            self.pool(hidden_state, sequence_lengths=sequence_lengths, n_tokens=n_tokens)
            for hidden_state in out.hidden_states[1:])

        return pooled_hidden_states if index is None else pooled_hidden_states[index]
    
    def pool(self, hidden_state, sequence_lengths, n_tokens):
        result = []
        batch_size = hidden_state.shape[0]
        for i in range(n_tokens):
            seq_len = sequence_lengths + i
            seq_len.to(hidden_state.device)
            result.append(
                hidden_state[torch.arange(batch_size, device=hidden_state.device), seq_len]
            )
        result = torch.stack(result) # N TOKEN, N BATCH, DIM
        result = result.transpose(0,1)
        result = result.reshape(batch_size, -1) # N BATCH , N TOKEN * DIM
        return result
    
    def prepare_tokenizer(self, tokenizer):
        # Add padding
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.config.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "left"

        # Add KG special token
        self.config.kg_token = KG_MASK
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

### GPT2
class GPT2ForLMKBC(LanguageModelForLMKBC, GPT2LMHeadModel):
    @property
    def backbone(self):
        return self.transformer
    
    @property
    def embeddings(self):
        return self.transformer.wte
### LLAMA
class LlamaForLMKBC(LanguageModelForLMKBC, LlamaForCausalLM):
    @property
    def backbone(self):
        return self.model
    
    @property
    def embeddings(self):
        return self.model.embed_tokens
### MISTRAL
class MistralForLMKBC(LanguageModelForLMKBC, MistralForCausalLM):
    @property
    def backbone(self):
        return self.model
    
    @property
    def embeddings(self):
        return self.model.embed_tokens
### OPT
class OPTForLMKBC(LanguageModelForLMKBC, OPTForCausalLM):
    @property
    def backbone(self):
        return self.model.decoder
    
    @property
    def embeddings(self):
        return self.model.decoder.embed_tokens

### AUTO
MAPPING = {
    "gpt2" : GPT2ForLMKBC,
    "llama" : LlamaForLMKBC,
    "mistral" : MistralForLMKBC,
    "opt" : OPTForLMKBC
}

class AutoModelForLMKBC:
    @staticmethod
    def from_pretrained(model_name_or_path, **kwargs) -> LanguageModelForLMKBC:
        config = AutoConfig.from_pretrained(model_name_or_path)
        constructor = MAPPING[config.model_type]
        return constructor.from_pretrained(model_name_or_path, **kwargs)