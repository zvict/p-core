import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils import LayerNorm


class TransformerEncoderLayer(nn.Module):
    """
    Custom TransformerEncoderLayer that allows optional LayerNorm.
    
    Based on PyTorch's TransformerEncoderLayer but with configurable normalization.
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation=F.relu, batch_first=False, norm_first=False,
                 use_norm=True, norm_eps=1e-5, use_xavier_init=False, init_gain=1.0):
        """
        Args:
            d_model: The number of expected features in the input (required).
            nhead: The number of heads in the multiheadattention models (required).
            dim_feedforward: The dimension of the feedforward network model (default=2048).
            dropout: The dropout value (default=0.1).
            activation: The activation function of intermediate layer, relu or gelu (default=relu).
            batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) (default=False).
            norm_first: If True, layer norm is done prior to attention and feedforward operations, respectively. Otherwise it's done after (default=False).
            use_norm: If True, use LayerNorm. If False, skip normalization entirely (default=True).
            norm_eps: A value added to the denominator for numerical stability (default=1e-5).
            use_xavier_init: If True, initialize linear layers with xavier_uniform_. If False, use PyTorch default initialization (default=False).
            init_gain: Gain parameter for xavier_uniform_ initialization (default=1.0). Only used if use_xavier_init=True.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.use_norm = use_norm
        self.norm_first = norm_first
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Optional xavier_uniform_ initialization
        if use_xavier_init:
            nn.init.xavier_uniform_(self.linear1.weight, gain=init_gain)
            nn.init.xavier_uniform_(self.linear2.weight, gain=init_gain)
            
            # Initialize biases to zero (standard practice)
            if self.linear1.bias is not None:
                nn.init.constant_(self.linear1.bias, 0.0)
            if self.linear2.bias is not None:
                nn.init.constant_(self.linear2.bias, 0.0)
        
        # Normalization layers (optional)
        if self.use_norm:
            self.norm1 = LayerNorm(d_model, eps=norm_eps)
            self.norm2 = LayerNorm(d_model, eps=norm_eps)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = activation
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: The sequence to the encoder layer (required).
            src_mask: The mask for the src sequence (optional).
            src_key_padding_mask: The mask for the src keys per batch (optional).
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model) if batch_first=True, 
            or (seq_len, batch, d_model) if batch_first=False.
        """
        x = src
        
        if self.norm_first:
            # Pre-norm architecture
            x2 = self.norm1(x)
            x2 = self.self_attn(x2, x2, x2, attn_mask=src_mask, 
                               key_padding_mask=src_key_padding_mask)[0]
            x = x + self.dropout1(x2)
            
            x2 = self.norm2(x)
            x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
            x = x + self.dropout2(x2)
        else:
            # Post-norm architecture (default)
            x2 = self.self_attn(x, x, x, attn_mask=src_mask, 
                               key_padding_mask=src_key_padding_mask)[0]
            x = x + self.dropout1(x2)
            x = self.norm1(x)
            
            x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = x + self.dropout2(x2)
            x = self.norm2(x)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Custom TransformerEncoder that stacks multiple TransformerEncoderLayer instances.
    
    Based on PyTorch's TransformerEncoder but works with our custom TransformerEncoderLayer.
    """
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        """
        Args:
            encoder_layer: An instance of TransformerEncoderLayer (required).
            num_layers: The number of sub-encoder layers in the encoder (required).
            norm: The layer normalization component (optional). If None, no normalization is applied at the end.
        """
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        if norm is not None:
            self.norm = norm
        else:
            self.norm = nn.Identity()
    
    @torch.compile
    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Args:
            src: The sequence to the encoder (required).
            mask: The mask for the src sequence (optional).
            src_key_padding_mask: The mask for the src keys per batch (optional).
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model) if batch_first=True, 
            or (seq_len, batch, d_model) if batch_first=False.
        """
        output = src
        
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        output = self.norm(output)
        
        return output


