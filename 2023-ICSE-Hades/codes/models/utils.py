import logging

import torch
from torch import nn
from torch.autograd import Variable
import math

# input: [batch_size, T, hidden_size * dir_num]
# input: [batch_size, T, encoded_dim * 2]
# output: [batch_size, 2 * hidden_size]
class Decoder(nn.Module):
    # input: [hidden_size, window_size]
    def __init__(self, encoded_dim, T, **kwargs):
        super(Decoder, self).__init__()
        linear_size = kwargs["linear_size"]

        layers = []
        # [(encoded_dim, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 2)]
        for i in range(kwargs["decoder_layer_num"] - 1):
            input_size = encoded_dim if i == 0 else linear_size
            layers += [nn.Linear(input_size, linear_size), nn.ReLU()]
        layers += [nn.Linear(linear_size, 2)]
        self.net = nn.Sequential(*layers)

        self.self_attention = kwargs["self_attention"]
        if self.self_attention:
            self.attn = SelfAttention(encoded_dim, T) # T时间步

    # input: [batch_size, T, hidden_size * dir_num]
    # input: [batch_size, T, encoded_dim * 2]
    # output: [batch_size, 2 * hidden_size]
    def forward(self, x: torch.Tensor):
        if self.self_attention:
            ret = self.attn(x)
        else:
            ret = x[:, -1, :]
        return self.net(ret)

# input: [batch_size, T, vocab_size]
# output: [batch_size, T, vocab_size, embedding_dim]
class Embedder(nn.Module):
    def __init__(self, vocab_size=300, **kwargs):
        super(Embedder, self).__init__()
        self.embedding_dim = kwargs["word_embedding_dim"]     # 32
        self.embedder = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)

    # input: [batch_size, T, vocab_size]
    # output: [batch_size, T, vocab_size, embedding_dim]
    def forward(self, x):
        return self.embedder(x.long())

# input: [batch_size, T, encoded_dim * 2]
# output: [batch_size, encoded_dim * 2]
class SelfAttention(nn.Module):
    # input: [encoded_dim, T]
    def __init__(self, input_size, seq_len):
        """
        Args:
            input_size: int, hidden_size * num_directions
            seq_len: window_size
        """
        super(SelfAttention, self).__init__()
        # [T, encoded_dim * 2, 1]
        self.atten_w = nn.Parameter(torch.randn(seq_len, input_size, 1))
        # [T, 1, 1]
        self.atten_bias = nn.Parameter(torch.randn(seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.atten_bias.data.fill_(0)

    # input: [batch_size, T, encoded_dim * 2]
    # output: [batch_size, encoded_dim * 2]
    def forward(self, x):
        # x: [batch_size, window_size, 2 * hidden_size]
        # x: [batch_size, T, encoded_dim * 2]
        # Exchange two dimensions
        input_tensor = x.transpose(1, 0)  # w x b x 2h
        # w, b, 2h; w, 2h, 1
        # logging.info(input_tensor.shape, self.atten_w.shape)
        input_tensor = (torch.bmm(input_tensor, self.atten_w) + self.atten_bias)  # w x b x 1
        # b x w x 1
        input_tensor = input_tensor.transpose(1, 0)
        # b x w x 1
        atten_weight = input_tensor.tanh()
        # b, 1, w; b, w, 2h -> b, 1, 2h --squeeze-> b, 2h
        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), x).squeeze()
        return weighted_sum

    # Glorot initial is also called xavier initial, sqrt(6 / (fan_in + fan_out ))
    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)


class Trans(nn.Module):
    def __init__(self, input_size, layer_num, out_dim, dim_feedforward=512, dropout=0, device="cpu", norm=None,
                 nhead=8):
        super(Trans, self).__init__()
        # TransformerEncoderLayer and TransformerEncoder don't change the dimension
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, # the dimension of input
            dim_feedforward=dim_feedforward,  # the dimension of hidden layer default: 2048
            nhead=nhead, # the number of heads in the multiheadattention models
            dropout=dropout,
            batch_first=True)
        self.net = nn.TransformerEncoder(encoder_layer, num_layers=layer_num, norm=norm).to(device)
        self.out_layer = nn.Linear(input_size, out_dim)

    # input: [batch_size, T, var] [batch_size, T, input_size]
    # output: [batch_size, T, out_dim]
    def forward(self, x: torch.Tensor):
        # out: [batch_size, T, input_size]
        out = self.net(x)
        return self.out_layer(out)
