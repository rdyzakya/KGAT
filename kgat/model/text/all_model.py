from transformers import (
    GPT2LMHeadModel,
    LlamaForCausalLM,
    MistralForCausalLM
)
from .abstract import LMKBCWrapper

catalog = [
    "MistralForLMKBC",
    "LlamaForLMKBC",
    "GPT2ForLMKBC",
]

class GPT2ForLMKBC(LMKBCWrapper, GPT2LMHeadModel):
    @property
    def backbone(self):
        return self.transformer
    
    @property
    def embeddings(self):
        return self.transformer.wte
    
class LlamaForLMKBC(LMKBCWrapper, LlamaForCausalLM):
    @property
    def backbone(self):
        return self.model
    
    @property
    def embeddings(self):
        return self.model.embed_tokens
    
class MistralForLMKBC(LMKBCWrapper, MistralForCausalLM):
    @property
    def backbone(self):
        return self.model
    
    @property
    def embeddings(self):
        return self.model.embed_tokens