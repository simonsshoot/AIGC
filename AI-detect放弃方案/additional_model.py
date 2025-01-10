# import torch
# import torch.nn as nn
# from custom_LSTM import CustomRNN

# class GCN(nn.Module):
#     def __init__(self, GCN_layer, input_size, relation_n):
#         super(GCN, self).__init__()
#         self.GCN_layer = GCN_layer
#         self.relation_n = relation_n
#         self.GCNweight = nn.ModuleList()

#         if relation_n == 0:
#             relation_n = 1
#         for _ in range(GCN_layer * relation_n):
#             self.GCNweight.append(nn.Linear(input_size, input_size))

#     def normalize_laplacian_matrix(self, adj):
#         row_sum_invSqrt, temp = torch.pow(adj.sum(2) + 1e-30, -0.5), []
#         for i in range(adj.size()[0]):
#             temp.append(torch.diag(row_sum_invSqrt[i]))
#         degree_matrix = torch.cat(temp, dim=0).view(adj.size())
#         return torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix).to(
#             degree_matrix.device
#         )

#     def forward(self, nodes_rep, adj_metric):
#         relation_num = self.relation_n
#         if relation_num == 0:
#             normalized_laplacian_matrix = self.normalize_laplacian_matrix(adj_metric)
#             normalized_laplacian_matrix.requires_grad = False
#             nodes_rep_history = [nodes_rep]
#             for i in range(self.GCN_layer):
#                 tmp_rep = torch.bmm(normalized_laplacian_matrix, nodes_rep_history[i])
#                 nodes_rep_history.append(torch.tanh(self.GCNweight[i](tmp_rep)))
#             nodes_rep_history = torch.stack(nodes_rep_history, dim=0)
#             return nodes_rep_history
#         else:
#             adj_idx = 0
#             tot_nodes_rep_history = list()
#             adj_metric = adj_metric.transpose(0, 1)
#             for adj in adj_metric:
#                 normalized_laplacian_matrix = self.normalize_laplacian_matrix(adj)
#                 normalized_laplacian_matrix.requires_grad = False
#                 nodes_rep_history = [nodes_rep]
#                 for i in range(self.GCN_layer):
#                     tmp_rep = torch.bmm(
#                         normalized_laplacian_matrix, nodes_rep_history[i]
#                     )
#                     nodes_rep_history.append(
#                         torch.tanh(
#                             self.GCNweight[adj_idx * self.GCN_layer + i](tmp_rep)
#                         )
#                     )
#                 nodes_rep_history = torch.stack(nodes_rep_history, dim=0)
#                 tot_nodes_rep_history.append(nodes_rep_history)
#                 adj_idx += 1
#             tot_nodes_rep_history = torch.stack(tot_nodes_rep_history, axis=0)
#             tot_nodes_rep_history = torch.sum(tot_nodes_rep_history, axis=0)
#             return tot_nodes_rep_history

# def function_align(x, y, x_mask, y_mask, input_size):
#     x_mask_tile = x_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])
#     y_mask_tile = y_mask.unsqueeze(-1).repeat(1, 1, y.shape[-1])
#     x = x * x_mask_tile.float()
#     y = y * y_mask_tile.float()
#     return torch.cat([x - y, x * y], dim=2)

# def mask_mean(x, x_mask, dim):
#     """
#     :param x: batch, nodes_num, hidden_size
#     :param x_mask: batch, nodes_num
#     :param dim:
#     :return: x
#     """
#     x_mask_tile = x_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])
#     assert x.shape == x_mask_tile.shape, "x shape {}, x_mask_tile shape {}".format(
#         x.shape, x_mask_tile.shape
#     )

#     result = torch.sum(x * x_mask_tile.float(), dim=dim) / (
#         torch.sum(x_mask_tile.float(), dim=dim) + 1e-30
#     )

#     return result

# class GCNGraphAgg(nn.Module):
#     def __init__(
#         self,
#         input_size,
#         max_sentence_num,
#         gcn_layer,
#         node_size,
#         attention_maxscore=None,
#         relation_num=None,
#     ):
#         super(GCNGraphAgg, self).__init__()
#         self.input_size = input_size
#         self.max_sentence_num = max_sentence_num
#         self.gcn_layer = gcn_layer
#         self.node_size = node_size
#         self.relation_num = relation_num
#         self.graph_node_proj = nn.Linear(input_size, node_size)
#         self.align_proj = nn.Linear(self.node_size * 2, self.node_size)
#         self.GCN = GCN(self.gcn_layer, self.node_size, self.relation_num)
#         self.rnn_coherence_proj = CustomRNN(
#             input_size=self.node_size,
#             hidden_size=self.node_size,
#             batch_first=True,
#             max_seq_length=max_sentence_num,
#             attention_maxscore=attention_maxscore,
#         )

#     def forward(
#         self,
#         hidden_states,
#         nodes_index_mask,
#         adj_metric,
#         node_mask,
#         sen2node,
#         sentence_mask,
#         sentence_length,
#     ):
#         """
#         :param hidden_states: batch, seq_len, hidden_size
#         :param nodes_mask: batch, node_num, seq_len
#         :param claim_node_mask: batch, claim_node_num, seq_len
#         :return: logits
#         """
#         """evidence nodes and edges presentation"""
#         nodes_rep = torch.bmm(nodes_index_mask, hidden_states)
#         nodes_rep = torch.relu(self.graph_node_proj(nodes_rep))

#         """GCN propagation"""
#         nodes_rep_history = self.GCN(nodes_rep, adj_metric)
#         joint_nodes_rep = nodes_rep_history[-1, :, :, :]
#         sens_rep = torch.bmm(sen2node, joint_nodes_rep)

#         final_rep, padded_output = self.rnn_coherence_proj(
#             sens_rep, sentence_length, sentence_mask
#         )

#         return final_rep
    
# Super Graph!!

# import torch
# import torch.nn as nn
# from custom_LSTM import CustomRNN

# class GCN(nn.Module):
#     def __init__(self, GCN_layer, input_size, relation_n):
#         super(GCN, self).__init__()
#         self.GCN_layer = GCN_layer
#         self.relation_n = relation_n
#         self.GCNweight = nn.ModuleList()

#         if relation_n == 0:
#             relation_n = 1
#         for _ in range(GCN_layer * relation_n):
#             self.GCNweight.append(nn.Linear(input_size, input_size))

#     def normalize_laplacian_matrix(self, adj):
#         row_sum_invSqrt, temp = torch.pow(adj.sum(2) + 1e-30, -0.5), []
#         for i in range(adj.size()[0]):
#             temp.append(torch.diag(row_sum_invSqrt[i]))
#         degree_matrix = torch.cat(temp, dim=0).view(adj.size())
#         return torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix).to(
#             degree_matrix.device
#         )

#     def forward(self, nodes_rep, adj_metric):
#         relation_num = self.relation_n
#         if relation_num == 0:
#             normalized_laplacian_matrix = self.normalize_laplacian_matrix(adj_metric)
#             normalized_laplacian_matrix.requires_grad = False
#             nodes_rep_history = [nodes_rep]
#             for i in range(self.GCN_layer):
#                 tmp_rep = torch.bmm(normalized_laplacian_matrix, nodes_rep_history[i])
#                 nodes_rep_history.append(torch.tanh(self.GCNweight[i](tmp_rep)))
#             nodes_rep_history = torch.stack(nodes_rep_history, dim=0)
#             return nodes_rep_history
#         else:
#             adj_idx = 0
#             tot_nodes_rep_history = []
#             adj_metric = adj_metric.transpose(0, 1)  # [relation_num, batch, max_nodes_num, max_nodes_num]
#             for adj in adj_metric:
#                 normalized_laplacian_matrix = self.normalize_laplacian_matrix(adj)
#                 normalized_laplacian_matrix.requires_grad = False
#                 nodes_rep_history = [nodes_rep]
#                 for i in range(self.GCN_layer):
#                     tmp_rep = torch.bmm(
#                         normalized_laplacian_matrix, nodes_rep_history[i]
#                     )
#                     nodes_rep_history.append(
#                         torch.tanh(
#                             self.GCNweight[adj_idx * self.GCN_layer + i](tmp_rep)
#                         )
#                     )
#                 nodes_rep_history = torch.stack(nodes_rep_history, dim=0)
#                 tot_nodes_rep_history.append(nodes_rep_history)
#                 adj_idx += 1
#             tot_nodes_rep_history = torch.stack(tot_nodes_rep_history, dim=0)
#             tot_nodes_rep_history = torch.sum(tot_nodes_rep_history, dim=0)
#             return tot_nodes_rep_history

# def function_align(x, y, x_mask, y_mask, input_size):
#     x_mask_tile = x_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])
#     y_mask_tile = y_mask.unsqueeze(-1).repeat(1, 1, y.shape[-1])
#     x = x * x_mask_tile.float()
#     y = y * y_mask_tile.float()
#     return torch.cat([x - y, x * y], dim=2)

# def mask_mean(x, x_mask, dim):
#     """
#     :param x: batch, nodes_num, hidden_size
#     :param x_mask: batch, nodes_num
#     :param dim:
#     :return: x
#     """
#     x_mask_tile = x_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])
#     assert x.shape == x_mask_tile.shape, "x shape {}, x_mask_tile shape {}".format(
#         x.shape, x_mask_tile.shape
#     )

#     result = torch.sum(x * x_mask_tile.float(), dim=dim) / (
#         torch.sum(x_mask_tile.float(), dim=dim) + 1e-30
#     )

#     return result

# class GCNGraphAgg(nn.Module):
#     def __init__(
#         self,
#         input_size,
#         max_sentence_num,  # 应该与 args.max_sentences 一致
#         gcn_layer,
#         node_size,
#         attention_maxscore=None,
#         relation_num=None,
#     ):
#         super(GCNGraphAgg, self).__init__()
#         self.input_size = input_size
#         self.max_sentence_num = max_sentence_num
        
#         self.gcn_layer = gcn_layer
#         self.node_size = node_size
#         self.relation_num = relation_num
#         self.graph_node_proj = nn.Linear(input_size, node_size)
#         self.align_proj = nn.Linear(self.node_size * 2, self.node_size)
#         self.GCN = GCN(self.gcn_layer, self.node_size, self.relation_num)
#         self.rnn_coherence_proj = CustomRNN(
#             input_size=self.node_size,
#             hidden_size=self.node_size,
#             batch_first=True,
#             max_seq_length=max_sentence_num,
#             attention_maxscore=attention_maxscore,
#         )
        
#     def forward(
#         self,
#         hidden_states,
#         nodes_index_mask,
#         adj_metric,
#         node_mask,
#         sen2node,
#         sentence_mask,
#         sentence_length,
#         hyperedges=None,  # 新增超边输入
#     ):
#         # Step 1: Compute node representations
#         nodes_rep = torch.bmm(nodes_index_mask, hidden_states)  # [batch, max_nodes_num, hidden_size]
#         nodes_rep = torch.relu(self.graph_node_proj(nodes_rep))  # [batch, max_nodes_num, node_size]
#         # 

#         # Step 2: GCN propagation
#         nodes_rep_history = self.GCN(nodes_rep, adj_metric)  # List of [layers, batch, max_nodes_num, node_size]
#         joint_nodes_rep = nodes_rep_history[-1]  # [batch, max_nodes_num, node_size]
        

#         # Step 3: Aggregate node representations into sentence representations
#         sens_rep = torch.bmm(sen2node, joint_nodes_rep)  # [batch, max_sentences, node_size]
        

#         # Step 4: Process hyperedges if provided
#         if hyperedges is not None and hyperedges.dim() == 3 and hyperedges.size(1) > 0:
            
#             # Compute hyperedge mask
#             hyperedge_mask = (hyperedges.sum(dim=2) > 0).float()  # [batch, max_hyperedges]
            

#             # Aggregate node representations for each hyperedge
#             hyperedge_rep = torch.bmm(hyperedges, joint_nodes_rep) / (
#                 hyperedges.sum(dim=2, keepdim=True) + 1e-6
#             )  # [batch, max_hyperedges, node_size]
      

#             # Compute mean of hyperedge representations
#             hyperedge_rep = mask_mean(hyperedge_rep, hyperedge_mask, dim=1)  # [batch, node_size]
           

#             # Expand hyperedge_rep to match the number of sentences
#             hyperedge_rep_expanded = hyperedge_rep.unsqueeze(1).expand(-1, sens_rep.size(1), -1)  # [batch, max_sentences, node_size]
           

#             # Align and combine sentence representations with hyperedge representations
#             combined_rep = function_align(
#                 sens_rep, 
#                 hyperedge_rep_expanded, 
#                 sentence_mask, 
#                 sentence_mask,  # 使用与 x_mask 相同的 mask
#                 self.node_size
#             )  # [batch, max_sentences, node_size * 2]
           

#             # Project back to node_size
#             combined_rep = self.align_proj(combined_rep)  # [batch, max_sentences, node_size]
           
#         else:
       
#             combined_rep = sens_rep  # [batch, max_sentences, node_size]

#         # Step 5: Pass through RNN for coherence
#         final_rep, padded_output = self.rnn_coherence_proj(
#             combined_rep, sentence_length, sentence_mask
#         )  # final_rep: [batch, node_size]
   

#         return final_rep

# GAT!!

import torch
import torch.nn as nn
from custom_LSTM import CustomRNN
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(GATLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.attn_weight = nn.Parameter(torch.Tensor(input_size, output_size))
        self.attn_bias = nn.Parameter(torch.Tensor(output_size))
        self.attn_alpha = nn.Parameter(torch.Tensor(1))

        nn.init.xavier_uniform_(self.attn_weight)
        nn.init.zeros_(self.attn_bias)
        nn.init.zeros_(self.attn_alpha)

    def forward(self, nodes_rep, adj_metric):
        # 计算注意力系数
        h = torch.matmul(nodes_rep, self.attn_weight)
        attention_scores = torch.matmul(h, h.transpose(1, 2))  # 计算节点之间的相似度
        attention_scores = F.leaky_relu(attention_scores, negative_slope=0.2)
        
        # 使用邻接矩阵来筛选出有效的边
        attention_scores = attention_scores * adj_metric  # 仅保留有效连接的attention值
        attention_scores = torch.softmax(attention_scores, dim=-1)  # 对邻接关系进行softmax归一化

        # 进行加权求和
        output = torch.matmul(attention_scores, nodes_rep)
        output = output + self.attn_bias  # 加上偏置
        return output

class GCN(nn.Module):
    def __init__(self, GCN_layer, input_size, relation_n):
        super(GCN, self).__init__()
        self.GCN_layer = GCN_layer
        self.relation_n = relation_n
        self.GATweight = nn.ModuleList()

        if relation_n == 0:
            relation_n = 1
        for _ in range(GCN_layer * relation_n):
            self.GATweight.append(GATLayer(input_size, input_size))

    def normalize_laplacian_matrix(self, adj):
        row_sum_invSqrt, temp = torch.pow(adj.sum(2) + 1e-30, -0.5), []
        for i in range(adj.size()[0]):
            temp.append(torch.diag(row_sum_invSqrt[i]))
        degree_matrix = torch.cat(temp, dim=0).view(adj.size())
        return torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix).to(
            degree_matrix.device
        )

    def forward(self, nodes_rep, adj_metric):
        relation_num = self.relation_n
        if relation_num == 0:
            normalized_laplacian_matrix = self.normalize_laplacian_matrix(adj_metric)
            normalized_laplacian_matrix.requires_grad = False
            nodes_rep_history = [nodes_rep]
            for i in range(self.GCN_layer):
                tmp_rep = torch.bmm(normalized_laplacian_matrix, nodes_rep_history[i])
                nodes_rep_history.append(torch.tanh(self.GATweight[i](tmp_rep, adj_metric)))  # 使用GAT层
            nodes_rep_history = torch.stack(nodes_rep_history, dim=0)
            return nodes_rep_history
        else:
            adj_idx = 0
            tot_nodes_rep_history = list()
            adj_metric = adj_metric.transpose(0, 1)
            for adj in adj_metric:
                normalized_laplacian_matrix = self.normalize_laplacian_matrix(adj)
                normalized_laplacian_matrix.requires_grad = False
                nodes_rep_history = [nodes_rep]
                for i in range(self.GCN_layer):
                    tmp_rep = torch.bmm(
                        normalized_laplacian_matrix, nodes_rep_history[i]
                    )
                    nodes_rep_history.append(
                        torch.tanh(
                            self.GATweight[adj_idx * self.GCN_layer + i](tmp_rep, adj)  # 使用GAT层
                        )
                    )
                nodes_rep_history = torch.stack(nodes_rep_history, dim=0)
                tot_nodes_rep_history.append(nodes_rep_history)
                adj_idx += 1
            tot_nodes_rep_history = torch.stack(tot_nodes_rep_history, axis=0)
            tot_nodes_rep_history = torch.sum(tot_nodes_rep_history, axis=0)
            return tot_nodes_rep_history

def function_align(x, y, x_mask, y_mask, input_size):
    x_mask_tile = x_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])
    y_mask_tile = y_mask.unsqueeze(-1).repeat(1, 1, y.shape[-1])
    x = x * x_mask_tile.float()
    y = y * y_mask_tile.float()
    return torch.cat([x - y, x * y], dim=2)

def mask_mean(x, x_mask, dim):
    """
    :param x: batch, nodes_num, hidden_size
    :param x_mask: batch, nodes_num
    :param dim:
    :return: x
    """
    x_mask_tile = x_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])
    assert x.shape == x_mask_tile.shape, "x shape {}, x_mask_tile shape {}".format(
        x.shape, x_mask_tile.shape
    )

    result = torch.sum(x * x_mask_tile.float(), dim=dim) / (
        torch.sum(x_mask_tile.float(), dim=dim) + 1e-30
    )

    return result

class GCNGraphAgg(nn.Module):
    def __init__(
        self,
        input_size,
        max_sentence_num,
        gcn_layer,
        node_size,
        attention_maxscore=None,
        relation_num=None,
    ):
        super(GCNGraphAgg, self).__init__()
        self.input_size = input_size
        self.max_sentence_num = max_sentence_num
        self.gcn_layer = gcn_layer
        self.node_size = node_size
        self.relation_num = relation_num
        self.graph_node_proj = nn.Linear(input_size, node_size)
        self.align_proj = nn.Linear(self.node_size * 2, self.node_size)
        self.GCN = GCN(self.gcn_layer, self.node_size, self.relation_num)
        self.rnn_coherence_proj = CustomRNN(
            input_size=self.node_size,
            hidden_size=self.node_size,
            batch_first=True,
            max_seq_length=max_sentence_num,
            attention_maxscore=attention_maxscore,
        )

    def forward(
        self,
        hidden_states,
        nodes_index_mask,
        adj_metric,
        node_mask,
        sen2node,
        sentence_mask,
        sentence_length,
    ):
        """
        :param hidden_states: batch, seq_len, hidden_size
        :param nodes_mask: batch, node_num, seq_len
        :param claim_node_mask: batch, claim_node_num, seq_len
        :return: logits
        """
        """evidence nodes and edges presentation"""
        nodes_rep = torch.bmm(nodes_index_mask, hidden_states)
        nodes_rep = torch.relu(self.graph_node_proj(nodes_rep))

        """GCN propagation with attention"""
        nodes_rep_history = self.GCN(nodes_rep, adj_metric)
        joint_nodes_rep = nodes_rep_history[-1, :, :, :]
        sens_rep = torch.bmm(sen2node, joint_nodes_rep)

        final_rep, padded_output = self.rnn_coherence_proj(
            sens_rep, sentence_length, sentence_mask
        )

        return final_rep
