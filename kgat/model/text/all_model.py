from transformers import (
    GPT2LMHeadModel,
    LlamaForCausalLM,
    MistralForCausalLM
)
from .abstract import LMKBCWrapper

catalog = {
    "MistralForLMKBC" : ["mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.3"],
    "LlamaForLMKBC" : ["meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7b-hf"],
    "GPT2ForLMKBC" : ["openai-community/gpt2", "gpt2"],
}

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