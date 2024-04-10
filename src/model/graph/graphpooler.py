import torch
import torch_geometric

class VirtualNodeGraphPooler(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def add_virtual_node(self, x, edge_index, edge_attribute, batch):
        unique_batch = torch.unique(batch)
        new_x = [x]
        new_edge_index = [edge_index]
        new_edge_attribute = [edge_attribute]

        virtual_node_id = x.shape[0]
        for b in unique_batch:
            new_x.append(torch.zeros_like(x[0]))

            head_index = torch.argwhere(batch == b)
            added_edge_index = torch.hstack([
                head_index,
                torch.full(head_index.shape, virtual_node_id)
            ])
            new_edge_index.append(added_edge_index.T)

            new_edge_attribute.append(
                torch.zeros((added_edge_index.shape[0], edge_attribute.shape[1]))
            )

            virtual_node_id += 1

        x = torch.vstack(new_x)
        edge_index = torch.hstack(new_edge_index)
        edge_attribute = torch.vstack(new_edge_attribute)
        batch = torch.cat([batch, unique_batch])

        return x, edge_index, edge_attribute, batch


class GATCVirtualNodeGraphPooler(VirtualNodeGraphPooler):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, edge_dim):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.conv1 = torch_geometric.nn.conv.GATConv(in_channels,
                                  hidden_channels,
                                  heads=heads,
                                  edge_dim=edge_dim,
                                  add_self_loops=False,
                                  concat=False)
        self.conv2 = torch_geometric.nn.conv.GATConv(hidden_channels,
                                  out_channels,
                                  heads=heads,
                                  edge_dim=edge_dim,
                                  add_self_loops=False,
                                  concat=True)
        # self.linear = torch.nn.Linear(in_features=out_channels,
        #                               out_features=1,
        #                               bias=True)
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr, return_attention_weights=None)

        n_edge_index = edge_index.shape[1] # n_triplet
        n_node = x.shape[0]

        x, edge_index, edge_attr, batch = self.add_virtual_node(x, edge_index, edge_attr, batch)

        x, (_, att_weights) = self.conv2(x, edge_index, edge_attr, return_attention_weights=True)

        graph_emb = x[n_node:]
        att_weights = att_weights[:n_edge_index]
        batch = batch[:n_edge_index]
        edge_index = edge_index[:,:n_edge_index]

        # head_batch = batch[edge_index[0]]
        # tail_batch = batch[edge_index[1]]

        # if not (head_batch == tail_batch).all():
        #     raise Exception("There is an intersection between batch, make sure there are no edge connected between two or more batches")

        graph_emb = graph_emb.view(-1, self.heads, self.out_channels)

        # graph_emb2 = self.linear(graph_emb)
        # graph_emb2 = graph_emb2[head_batch]
        # graph_emb2 = graph_emb2.squeeze(-1)

        # pool_result = att_weights.log() * graph_emb2
        # pool_result = pool_result.mean(-1).sigmoid()

        return att_weights.log(), graph_emb