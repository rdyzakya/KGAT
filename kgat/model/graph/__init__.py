from .conv import (
    SigmoidGATConv
)
from .graphpooler import (
    GATCVirtualNodeGraphPooler,
    GATAggregateGraphPooler
)
from .subgraphpooler import (
    SubgraphPooler
)

import torch

class GraphModule(torch.nn.Module):
    def __init__(self, transformer, graphpooler, subgraphpooler, prepare_inputs_method, pad_token_id, batch_size=16):
        super().__init__()
        self.transformer = transformer
        self.graphpooler = graphpooler
        self.subgraphpooler = subgraphpooler
        self.prepare_inputs = prepare_inputs_method
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id

    def get_text_last_hidden_state(self, input_ids, attention_mask):
        prepared_inputs = self.prepare_inputs(input_ids=input_ids, attention_mask=attention_mask)
        # input_ids = prepared_inputs["input_ids"]
        # attention_mask = prepared_inputs["attention_mask"]
        # position_ids = prepared_inputs["position_ids"]
        hidden_states = self.transformer(**prepared_inputs)
        hidden_states = hidden_states[0]

        # get last token hidden state, as how any left-to-right model with seq-cls head do
        sequence_lengths = torch.eq(input_ids, self.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(hidden_states.device)

        batch_size = input_ids.shape[0]
        pooled_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

        return pooled_hidden_states
    
    def batch_text_last_hidden_state(self, input_ids, attention_mask):
        assert input_ids.shape == attention_mask.shape
        res = []

        for i in range(0, input_ids.shape[0], self.batch_size):
            current_inputs_ids = input_ids[i:i+self.batch_size]
            current_attention_mask = attention_mask[i:i+self.batch_size]

            current_hs = self.get_text_last_hidden_state(current_inputs_ids, current_attention_mask)

            res.append(current_hs)
        
        res = torch.vstack(res)

        return res

    
    def forward(self, graph_query_input_ids, graph_query_attention_mask,
                entities_input_ids, entities_attention_mask,
                relations_input_ids, relations_attention_mask,
                x_coo, batch):

        x = self.batch_text_last_hidden_state(entities_input_ids, entities_attention_mask)

        edge_attr = self.batch_text_last_hidden_state(relations_input_ids, relations_attention_mask)
        edge_attr = edge_attr[x_coo[1]]

        edge_index = x_coo[[True, False, True]]

        # RETRIEVE GRAPH EMB
        edge_score, graph_emb, edge_batch = self.graphpooler(x, edge_index, edge_attr, batch)
        # RETRIEVE SUBGRAPH EMB
        graph_query_emb = self.batch_text_last_hidden_state(graph_query_input_ids, graph_query_attention_mask)
        mean_fused_score, subgraph_emb, edge_batch = self.subgraphpooler(graph_query_emb, edge_score, graph_emb, edge_batch)

        return mean_fused_score, subgraph_emb, edge_batch
    
    def freeze_llm(self):
        # transformer
        for param in self.transformer.parameters():
            param.requires_grad = False
