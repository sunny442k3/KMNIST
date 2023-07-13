from typing import Optional
import sys
import torch
from torch import nn, Tensor
from torch.nn.modules.transformer import _get_activation_fn


class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(
            self, 
            tgt: Tensor, 
            memory: Tensor, 
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

@torch.jit.script
class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape)==3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class MLDecoder(nn.Module):
    def __init__(
            self, 
            num_classes=49, 
            n_heads=4,
            decoder_layers=1, 
            d_ff=2048, 
            decoder_embedding=512,
            hidden_dim=2048,
            query_weight=None):
        super(MLDecoder, self).__init__()

        
        embed_len_decoder = num_classes
        embed_standart = nn.Linear(hidden_dim, decoder_embedding)

        query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
        if query_weight is not None:
            query_embed.weight.data = torch.load(query_weight)
        query_embed.requires_grad_(False)

        decoder_dropout = 0.1
        # num_layers_decoder = 1
        # dim_feedforward = 2048
        layer_decode = TransformerDecoderLayerOptimal(
            d_model=decoder_embedding,
            nhead=n_heads,
            dim_feedforward=d_ff, 
            dropout=decoder_dropout
        )
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=decoder_layers)
        self.decoder.embed_standart = embed_standart
        self.decoder.query_embed = query_embed

        self.decoder.num_classes = num_classes
        self.decoder.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)
        self.decoder.duplicate_pooling = torch.nn.Parameter(
            torch.Tensor(embed_len_decoder, decoder_embedding, self.decoder.duplicate_factor))
        self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))

        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        self.decoder.group_fc = GroupFC(embed_len_decoder)
        self.train_wordvecs = None
        self.test_wordvecs = None
    
    def forward(self, x):
        if len(x.shape) == 4:
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:
            embedding_spatial = x
        embedding_spatial_786 = self.decoder.embed_standart(embedding_spatial)
        embedding_spatial_786 = torch.nn.functional.relu(embedding_spatial_786, inplace=True)
        bs = embedding_spatial_786.shape[0]
        query_embed = self.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1) 
        h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1)) 
        h = h.transpose(0, 1)
        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.decoder.duplicate_factor, device=h.device, dtype=h.dtype)
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)
        h_out = out_extrap.flatten(1)[:, :self.decoder.num_classes]
        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out
        return logits