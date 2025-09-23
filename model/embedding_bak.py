import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange
from model.attn_layer import (PositionalEmbedding,
                              AttentionLayer,
                              complex_dropout,
                              complex_operator)
from model.Conv_Blocks import Inception_Block
from model.multi_attention_blocks import Inception_Attention_Block
from model.DualSTF import DualSTBlock

from model.RevIN import RevIN

class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()

        self.attn_layer = attn
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):


        out, attn = self.attn_layer(x)
        y = complex_dropout(self.dropout, out)

        return y



class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):

        for attn_layer in self.attn_layers:
            x, _ = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


class TokenEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, n_window=100, n_layers=1, branch_layers=['fc_linear', 'intra_fc_transformer'],
                 group_embedding='False', match_dimension='first', kernel_size=[5], multiscale_patch_size=[10, 20],
                 init_type='normal', gain=0.02, dropout=0.1):
        super(TokenEmbedding, self).__init__()

        self.n_window = n_window
        self.window_size = n_window
        self.dropout = dropout
        self.branch_layers = branch_layers
        self.n_layers = n_layers
        self.match_dimension = match_dimension
        self.extended_dim = d_model
        self.multiscale_patch_size = multiscale_patch_size
        self.use_dual_block = False
        self.kernel_size = kernel_size
        self.in_dim = in_dim

        self.semantic_proj = nn.Linear(d_model, d_model)
        
        
        self.last_semantic_features = None
        
        self.encoder_layers = []
        self.norm_layers = []
        
        self.semantic_embeddings = []

        for i, e_layer in enumerate(branch_layers):
           
            extended_dim = self.extended_dim
            
            if e_layer == 'inter_fc_transformer':
                w_model = self.window_size // 2 + 1
                
                self.use_dual_block = True
                
                
                num_heads = 4  
                while extended_dim % num_heads != 0 and num_heads > 0:
                    num_heads -= 1
                
            
                if num_heads == 0:
                    num_heads = 1
                
                self.encoder_layers.append(DualSTBlock(
                    dim=extended_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    drop_path=dropout,
                    norm_layer=nn.LayerNorm
                ))
               
                self.semantic_embeddings.append(nn.Parameter(
                    torch.zeros(1, self.window_size // 10, extended_dim),
                    requires_grad=True
                ))
                
                self.norm_layers.append(nn.LayerNorm(extended_dim))

            elif e_layer == 'intra_fc_transformer':
                w_model = self.window_size // 2 + 1
                
                self.use_dual_block = True
                
               
                num_heads = 4  
                while extended_dim % num_heads != 0 and num_heads > 0:
                    num_heads -= 1
                
          
                if num_heads == 0:
                    num_heads = 1
                
                self.encoder_layers.append(DualSTBlock(
                    dim=extended_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    drop_path=dropout,
                    norm_layer=nn.LayerNorm
                ))
               
                self.semantic_embeddings.append(nn.Parameter(
                    torch.zeros(1, self.window_size // 10, extended_dim),
                    requires_grad=True
                ))
              
                self.norm_layers.append(nn.LayerNorm(extended_dim))

            elif e_layer == 'multiscale_ts_attention':
                self.encoder_layers.append(Inception_Attention_Block(w_size=self.window_size,
                                                                     in_dim=extended_dim,
                                                                     d_model=extended_dim,
                                                                     patch_list=multiscale_patch_size))
           
                self.norm_layers.append(nn.Identity())
            else:
                raise ValueError(f'The specified model {e_layer} is not supported!')

        self.dropout = nn.Dropout(p=dropout)
        self.criterion = nn.MSELoss(reduction='none')
        self.activation = nn.GELU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)

                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):

                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        B, L, C = x.size()

        latent_list = []

        residual = None

        amplitudeRevIN = RevIN(int(L//2 + 1))

        dual_block_idx = 0

        last_semantics = None

        for i, (embedding_layer, norm_layer) in enumerate(zip(self.encoder_layers, self.norm_layers)):
            if self.branch_layers[i] not in ['linear', 'fc_linear', 'multiscale_ts_attention']:
                x = x.permute(0, 2, 1)

            if self.branch_layers[i] == 'multiscale_conv1d':
                x = complex_operator(embedding_layer, x)
            elif self.branch_layers[i] == 'multiscale_ts_attention':
               
                if last_semantics is not None:
               
                    projected_semantics = self.semantic_proj(last_semantics)
                    
                    if projected_semantics.shape[1] != x.shape[1]:
                        
                        temp_projected = torch.zeros((B, x.shape[1], projected_semantics.shape[2]), device=x.device)
                        for b in range(B):
                            for c in range(projected_semantics.shape[2]):
                                temp_projected[b, :, c] = F.interpolate(
                                    projected_semantics[b, :, c].unsqueeze(0).unsqueeze(0),
                                    size=x.shape[1],
                                    mode='linear',
                                    align_corners=False
                                ).squeeze()
                        projected_semantics = temp_projected
                    
                    
                    x = x + projected_semantics
                
                
                x = complex_operator(embedding_layer, x)
            elif self.branch_layers[i] in ['fc_linear']:
                x = torch.fft.rfft(x, dim=-2)
                x = complex_operator(embedding_layer, x)
                x = torch.fft.irfft(x, dim=-2)
            elif self.branch_layers[i] in ['inter_fc_transformer', 'intra_fc_transformer']:
                
                x = torch.fft.rfft(x, dim=-1)
                if self.branch_layers[i] == 'intra_fc_transformer':
                    x = x.permute(0, 2, 1)
                
                if not torch.is_complex(x):
                    
                    semantics = self.semantic_embeddings[dual_block_idx].expand(B, -1, -1)
                  
                    x, semantics_out = embedding_layer(x, semantics)
                  
                    last_semantics = semantics_out
                    dual_block_idx += 1
                else:
                   
                    semantics = self.semantic_embeddings[dual_block_idx].expand(B, -1, -1)
                    real_part, semantics_real = embedding_layer(x.real, semantics)
                    imag_part, semantics_imag = embedding_layer(x.imag, semantics)
                  
                    last_semantics = (semantics_real + semantics_imag) / 2
                    x = torch.complex(real_part, imag_part)
                    dual_block_idx += 1
                    
                if self.branch_layers[i] == 'intra_fc_transformer':
                    x = x.permute(0, 2, 1)
                x = torch.fft.irfft(x, dim=-1)
                
            
                latent_list.append(x)
               
                if residual is not None:
                    if x.shape == residual.shape and 'transformer' in self.branch_layers[i]:
                        x += residual
                residual = x
                
                continue
            else:
                x = complex_operator(embedding_layer, x)

            x = complex_operator(norm_layer, x)

          
            if self.branch_layers[i] not in ['linear', 'fc_linear', 'multiscale_ts_attention']:
                x = x.permute(0, 2, 1)

            latent_list.append(x)

    
            if residual is not None:
                if x.shape == residual.shape and 'transformer' in self.branch_layers[i]:
                    x += residual

            residual = x

        self.last_semantic_features = last_semantics
        
        return x, latent_list

class InputEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, n_window, device, dropout=0.1, n_layers=1, use_pos_embedding='False',
                 group_embedding='False', kernel_size=5, init_type='kaiming', match_dimension='first',  branch_layers=['linear']):
        super(InputEmbedding, self).__init__()
        self.device = device
        self.token_embedding = TokenEmbedding(in_dim=in_dim, d_model=d_model, n_window=n_window,
                                              n_layers=n_layers, branch_layers=branch_layers,
                                              group_embedding=group_embedding, match_dimension=match_dimension,
                                              init_type=init_type, kernel_size=kernel_size,
                                              dropout=0.1)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.use_pos_embedding = use_pos_embedding
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.to(self.device)

        x, latent_list = self.token_embedding(x)

        if self.use_pos_embedding == 'True':
            x = x + self.pos_embedding(x).to(self.device)

        return self.dropout(x), latent_list

