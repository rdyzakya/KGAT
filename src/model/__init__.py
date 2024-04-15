import torch

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
            graph_query_input_ids, graph_query_attention_mask, graph_query_position_ids,
                entities_input_ids, entities_attention_mask, entities_position_ids,
                relations_input_ids, relations_attention_mask, relations_position_ids,
                x_coo, batch)
        
        out = self.text_module(prompt_input_ids, prompt_attention_mask, prompt_position_ids, subgraph_emb)

        return out, mean_fused_score, edge_batch