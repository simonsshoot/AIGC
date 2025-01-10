# import torch
# import torch.nn as nn
# import torch.nn.functional as f
# import numpy as np
# import math

# class CustomRNN(nn.Module):
#     def __init__(
#         self,
#         input_size,
#         hidden_size,
#         batch_first=True,
#         max_seq_length=60,
#         attention_maxscore=None,
#     ):
#         super(CustomRNN, self).__init__()
#         self.bidirect = False
#         self.num_layers = 1
#         self.num_heads = 1
#         self.batch_first = batch_first
#         self.with_weight = False
#         self.max_seq_length = max_seq_length
#         self.attention_maxscore = attention_maxscore
#         self.rnn = torch.nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             batch_first=batch_first,
#             bidirectional=self.bidirect,
#             num_layers=self.num_layers,
#         )
#         self.pooling = nn.AdaptiveMaxPool2d((1, input_size))

#     def forward(self, inputs, seq_lengths, sen_mask, method = "AttLSTM"):  
#         # input.size = (batch_size, max_seq_length, node_num)
#         # method can be "Pool", "LSTM", or 'AttLSTM"
#         if method == "LSTM":
#             packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
#                 inputs,
#                 seq_lengths.to("cpu"),
#                 batch_first=self.batch_first,
#                 enforce_sorted=False,
#             )
#             res, (hn, cn) = self.rnn(input=packed_inputs)
#             padded_res, _ = nn.utils.rnn.pad_packed_sequence(
#                 res, batch_first=self.batch_first, total_length=self.max_seq_length
#             )  
#             return hn.squeeze(0), padded_res
#         elif method == "AttLSTM":
#             sen_mask = sen_mask.to(inputs.device)  # 确保 sen_mask 在正确的设备上
#             att_inputs, att_inputs_weight = attention(
#             inputs,
#             inputs,
#             inputs,
#             sen_mask,
#             attention_maxscore=self.attention_maxscore,
#             )
#             packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
#                 att_inputs,
#                 seq_lengths.to("cpu"),
#                 batch_first=self.batch_first,
#                 enforce_sorted=False,
#             )
#             res, (hn, cn) = self.rnn(input=packed_inputs)
#             padded_res, _ = nn.utils.rnn.pad_packed_sequence(
#                 res, batch_first=self.batch_first, total_length=self.max_seq_length
#             )  
#             return hn.squeeze(0), padded_res
#         else:
#             out = self.pooling(inputs)
#             return out.squeeze(1), None


# def attention(query, key, value, mask=None, dropout=None, attention_maxscore=1000):
#     """Compute scaled dot product attention"""
#     d_k = query.size(-1)
#     query = f.normalize(query, p=2, dim=-1)
#     key = f.normalize(key, p=2, dim=-1)
#     scores = (
#         torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) * attention_maxscore
#     )
#     p_attn = None
#     if mask is not None:
#         for s, m in zip(scores, mask):
#             m = m.to(s.device)
#             s = s.masked_fill(m == 0, -1e9)
#             p = s.softmax(dim=-1)
#             if p_attn is None:
#                 p_attn = p
#             else:
#                 p_attn = torch.cat([p_attn, p], dim=0)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value), p_attn

#add tranformer!!
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CustomRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        max_seq_length=60,
        attention_maxscore=1000,
        num_transformer_layers=2,
        transformer_nhead=4,
        transformer_dim_feedforward=2048,
        transformer_dropout=0.1,
    ):
        super(CustomRNN, self).__init__()
        self.bidirect = False
        self.num_layers = 1
        self.num_heads = 1
        self.batch_first = batch_first
        self.with_weight = False
        self.max_seq_length = max_seq_length
        self.attention_maxscore = attention_maxscore
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=self.bidirect,
            num_layers=self.num_layers,
        )
        self.pooling = nn.AdaptiveMaxPool2d((1, input_size))
        
        # Initialize Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            activation='relu',
            batch_first=batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            norm=None  # You can add normalization if needed
        )
        self.transformer_hidden_size = hidden_size  # Adjust if necessary

    def forward(self, inputs, seq_lengths, sen_mask, method="Transformer"):
        # inputs.size = (batch_size, max_seq_length, input_size)
        if method == "LSTM":
            packed_inputs = nn.utils.rnn.pack_padded_sequence(
                inputs,
                seq_lengths.to("cpu"),
                batch_first=self.batch_first,
                enforce_sorted=False,
            )
            res, (hn, cn) = self.rnn(input=packed_inputs)
            padded_res, _ = nn.utils.rnn.pad_packed_sequence(
                res, batch_first=self.batch_first, total_length=self.max_seq_length
            )
            return hn.squeeze(0), padded_res

        elif method == "AttLSTM":
            sen_mask = sen_mask.to(inputs.device)  # Ensure sen_mask is on the correct device
            att_inputs, att_inputs_weight = attention(
                inputs,
                inputs,
                inputs,
                sen_mask,
                attention_maxscore=self.attention_maxscore,
            )
            packed_inputs = nn.utils.rnn.pack_padded_sequence(
                att_inputs,
                seq_lengths.to("cpu"),
                batch_first=self.batch_first,
                enforce_sorted=False,
            )
            res, (hn, cn) = self.rnn(input=packed_inputs)
            padded_res, _ = nn.utils.rnn.pad_packed_sequence(
                res, batch_first=self.batch_first, total_length=self.max_seq_length
            )
            return hn.squeeze(0), padded_res

        elif method == "Transformer":
            # Create attention mask for Transformer (True for positions to be masked)
            # Transformer expects mask shape: (batch_size, max_seq_length)
            # Convert sen_mask: 1 -> keep, 0 -> mask
            # So mask for Transformer should be (batch_size, max_seq_length)
            # with True where sen_mask == 0
            if sen_mask is not None:
                transformer_mask = ~sen_mask.bool()  # Invert mask: True where padding
            else:
                transformer_mask = None

            # Transformer expects inputs of shape (batch_size, max_seq_length, input_size) if batch_first=True
            transformer_out = self.transformer_encoder(inputs, src_key_padding_mask=transformer_mask)

            # To mimic LSTM's output, we can take the last hidden state based on seq_lengths
            # First, ensure transformer_out is (batch_size, max_seq_length, input_size)
            # Then gather the last valid output for each sequence
            if self.batch_first:
                batch_size, seq_len, embed_dim = transformer_out.size()
                # Create indices for the last valid time step for each sequence
                idx = (seq_lengths - 1).view(-1, 1).expand(-1, embed_dim).unsqueeze(1)  # (batch_size,1,embed_dim)
                hn = transformer_out.gather(1, idx).squeeze(1)  # (batch_size, embed_dim)
                padded_res = transformer_out  # (batch_size, max_seq_length, embed_dim)
            else:
                # If not batch_first, adjust accordingly
                transformer_out = transformer_out.transpose(0, 1)  # (batch_size, max_seq_length, embed_dim)
                batch_size, seq_len, embed_dim = transformer_out.size()
                idx = (seq_lengths - 1).view(-1, 1).expand(-1, embed_dim).unsqueeze(1)
                hn = transformer_out.gather(1, idx).squeeze(1)
                padded_res = transformer_out

            return hn, padded_res

        else:
            out = self.pooling(inputs)
            return out.squeeze(1), None

def attention(query, key, value, mask=None, dropout=None, attention_maxscore=1000):
    """Compute scaled dot product attention"""
    d_k = query.size(-1)
    query = F.normalize(query, p=2, dim=-1)
    key = F.normalize(key, p=2, dim=-1)
    scores = (
        torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) * attention_maxscore
    )
    p_attn = None
    if mask is not None:
        # Assuming mask is of shape (batch_size, max_seq_length)
        # Expand mask to match scores shape
        # scores: (batch_size, max_seq_length, max_seq_length)
        mask = mask.unsqueeze(1).expand(-1, scores.size(1), -1)  # (batch_size, max_seq_length, max_seq_length)
        scores = scores.masked_fill(~mask, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

