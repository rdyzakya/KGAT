from transformers import (
    GPT2LMHeadModel,
    LlamaForCausalLM,
    MistralForCausalLM
)
from abstract import LMKBCWrapper

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

if __name__ == "__main__":
    from transformers import AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    
    gpt2wrapper = GPT2ForLMKBC.from_pretrained("openai-community/gpt2")

    tokenizer = gpt2wrapper.prepare_tokenizer(tokenizer)

    text = ["Hello World", "What are you doing mate ? Are you lost ?"]
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    with torch.no_grad():
        out_transformer = gpt2wrapper.transformer(**inputs)
    logits_transformer = out_transformer[0]

    with torch.no_grad():
        out_wrapper = gpt2wrapper.last_hidden_state(**inputs)
    logits_wrapper = out_wrapper

    assert (logits_transformer[:,-1] == logits_wrapper).all()

    # LM KBC

    graph_embeddings = torch.randn(2, 3, gpt2wrapper.transformer.embed_dim)

    text = [f"According to the graph {gpt2wrapper.config.kg_token} then, S : Spongebob , P : Cooks , O :",
            f"Look at this graph : {gpt2wrapper.config.kg_token} we can infer that -> S : Spongebob , P : Cooks , O :"]
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    with torch.no_grad():
        out_lmkbc = gpt2wrapper.forward_lmkbc(graph_embeddings=graph_embeddings, **inputs)
    logits_lmkbc = out_lmkbc[0]

    assert logits_lmkbc.shape[0] == inputs["input_ids"].shape[0]
    assert logits_lmkbc.shape[1] == inputs["input_ids"].shape[1] + (graph_embeddings.shape[1] - 1)

    gpt2wrapper.train()
    for params in gpt2wrapper.parameters():
        assert params.requires_grad == False
    
    print(gpt2wrapper.embed_dim)