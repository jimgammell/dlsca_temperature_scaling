
# Adapted from https://github.com/suvadeep-iitb/TransNet/nn.Module):

from collections import OrderedDict
import numpy as np
import torch
from torch import nn

from models.common import VerboseModel

class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        demb
    ):
        super().__init__()
        
        self.inv_freq = 1 / (1e4 ** torch.range(0, demb, 2.0) / demb)
        
    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.einsum('i,j->ij', pos_seq, self.inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        if bsz is not None:
            return torch.tile(pos_emb.unsqueeze(1), (1, bsz, 1))
        else:
            return pos_emb.unsqueeze(1)

class Feedforward(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout=0.0
    ):
        super().__init__()
        
        self.dense_0 = nn.Linear(input_dim, hidden_dim)
        self.act_0 = nn.ReLU(inplace=True)
        self.dropout_0 = nn.Dropout(dropout)
        self.dense_1 = nn.Linear(hidden_dim, input_dim)
        self.dropout_1 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dense_0(x)
        x = self.act_0(x)
        x = self.dropout_0(x)
        x = self.dense_1(x)
        x = self.dropout_1(x)
        return x

class TransNet(VerboseModule):
    def __init__(
        self,
        input_shape,
        output_classes=256,
        depth,
        model_dims,
        n_heads,
        head_dims,
        hidden_dims,
        dropout=0.0,
        dropatt=0.0,
        embedding_conv_kernel_size=11,
        embedding_pool_size=11,
        clamp_len=-1,
        untie_r=False,
        smooth_pos_emb=True,
        untie_pos_emb=True,
        output_attn=False
    ):
        super().__init__()
        
        patch_embedding = nn.Sequential(
            nn.Conv1d(
                in_channels=input_shape[0], out_channels=model_dims, kernel_size=embedding_conv_kernel_size
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=embedding_pool_size, stride=embedding_pool_size)
        )
        
        if smooth_pos_emb:
            pos_emb = PositionalEmbedding(model_dims)
        else:
            assert clamp_len > 0
            if not untie_pos_emb:
                pos_emb = nn.Embedding(2*clamp_len+1, model_dims)
            else:
                pos_emb = None
        
        if not untie_r:
            self.register_parameter('r_w_bias', torch.zeros(n_heads, head_dims))
            self.register_parameter('r_r_bias', torch.zeros(n_heads, head_dims))
        
        transformer_layers = nn.Sequential(OrderedDict([
            ('layer_%d'%(layer_idx), TransformerLayer(
                n_head=n_head,
                model_dims=model_dims,
                head_dims=head_dims,
                hidden_dims=hidden_dims,
                dropout=dropout,
                dropatt=dropatt,
                r_w_bias=None if untie_r else self.r_w_bias,
                r_r_bias=None if untie_r else self.r_r_bias,
                smooth_pos_emb=smooth_pos_emb,
                untie_pos_emb=untie_pos_emb,
                clamp_len=clamp_len
            ))
        ]))
        
        global_average_pooling = nn.Sequential(
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        classifier = nn.Linear(model_dims, output_classes)
        
        self.model = nn.Sequential(
            patch_embedding,
            transformer_layers,
            global_average_pooling,
            classifier
        )
        
    def forward(self, x):
        return self.model(x)
        
