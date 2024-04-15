import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import graph
import text
from ..utils import Mask

class ModelForLMKBC(torch.nn.Module):
    def __init__(self, graph_module, text_module):
        super().__init__()
        self.graph_module = graph_module
        self.text_module = text_module
    
    def forward(self, graph_query_input_ids, graph_query_attention_mask,
                prompt_input_ids, prompt_attention_mask,
                entities_input_ids, entities_attention_mask,
                relations_input_ids, relations_attention_mask,
                x_coo, batch):
        
        mean_fused_score, subgraph_emb, edge_batch = self.graph_module(
            graph_query_input_ids, graph_query_attention_mask,
                entities_input_ids, entities_attention_mask,
                relations_input_ids, relations_attention_mask,
                x_coo, batch)
        
        out = self.text_module(prompt_input_ids, prompt_attention_mask, subgraph_emb)

        return out, mean_fused_score, edge_batch

def load_config(json_dict):
    # construct clm
    model_name_or_path = json_dict["clm"]["model_name_or_path"]
    clm = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_tokens(Mask.KG_MASK)

    # construct graph module
    ## prepare transformer
    transformer = eval(json_dict["graph_module"]["transformer"])
    ## construct graphpooler
    graphpooler_class = getattr(graph, json_dict["graph_module"]["graphpooler"]["class"])
    graphpooler_kwargs = json_dict["graph_module"]["graphpooler"]["kwargs"]
    graphpooler_kwargs["in_channels"] = eval(graphpooler_kwargs["in_channels"])
    graphpooler_kwargs["edge_dim"] = eval(graphpooler_kwargs["edge_dim"])
    graphpooler = graphpooler_class(**graphpooler_kwargs)
    ## construct subgraphpooler
    subgraphpooler_class = getattr(graph, json_dict["graph_module"]["subgraphpooler"]["class"])
    subgraphpooler_kwargs = json_dict["graph_module"]["subgraphpooler"]["kwargs"]
    subgraphpooler_kwargs["graph_emb_dim"] = graphpooler.out_channels
    subgraphpooler_kwargs["text_emb_dim"] = eval(subgraphpooler_kwargs["text_emb_dim"])
    subgraphpooler = subgraphpooler_class(**subgraphpooler_kwargs)

    graph_module = graph.GraphModule(transformer=transformer,
                                     graphpooler=graphpooler,
                                     subgraphpooler=subgraphpooler,
                                     prepare_inputs_method=clm.prepare_inputs_for_generation)
    
    # construct text module
    ## construc vt_transformer
    vt_transformer_class = getattr(text, json_dict["text_module"]["vt_transformer"]["class"])
    vt_transformer_kwargs = json_dict["text_module"]["vt_transformer"]["kwargs"]
    vt_transformer_kwargs["subgraph_emb_dim"] = graphpooler.out_channels
    vt_transformer_kwargs["word_emb_dim"] = eval(vt_transformer_kwargs["word_emb_dim"])
    vt_transformer = vt_transformer_class(**vt_transformer_kwargs)
    ## prepare clm.embedding
    clm_embedding = eval(json_dict["text_module"]["clm_embedding"])
    ## prepare clm
    clm_model = clm
    ## prepare kg_id
    kg_id = tokenizer.get_added_vocab()[Mask.KG_MASK]

    text_module = text.TextModule(vt_transformer=vt_transformer,
                                  clm_embedding=clm_embedding,
                                  clm_model=clm_model,
                                  kg_id=kg_id)
    
    model_for_lmkbc = ModelForLMKBC(graph_module=graph_module, text_module=text_module)

    return model_for_lmkbc, tokenizer