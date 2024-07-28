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
    
    def text_embedding(self, input_ids, attention_mask, index=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # hidden_states = out.hidden_states[index] # len(hidden_states) == self.config.n_layer + 1

        # get last token hidden state, as how any left-to-right model with seq-cls head do
        sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        # sequence_lengths = sequence_lengths.to(hidden_states.device)

        batch_size = input_ids.shape[0]
        # pooled_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        pooled_hidden_states = tuple(hidden_states[torch.arange(batch_size), sequence_lengths] for hidden_states in out.hidden_states[1:])

        return pooled_hidden_states if index is None else pooled_hidden_states[index]
    
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