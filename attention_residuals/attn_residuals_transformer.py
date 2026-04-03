"""Transformer model with Attention Residuals.

Implements a Transformer model that replaces standard residual
connections with Attention Residuals (AttnRes) as described in
"Attention Residuals" (Kimi Team, arXiv:2603.15031).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .attn_residuals import AttnRes


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention.
    """

    def __init__(self, 
            hidden_dim: int,
            num_heads: int,
            attn_dropout: float = 0.0,
            resid_dropout: float = 0.0,
            bias: bool = True,
            device = None,
            dtype = None
        ):
        super().__init__()        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        head_dim = hidden_dim // num_heads
        factory_kwargs = {"device": device, "dtype": dtype}         
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias, **factory_kwargs)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias, **factory_kwargs)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention for efficiency
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_dropout(self.o_proj(attn_out))


class Feedforward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
        activation="silu",
        device = None,
        dtype = None
    ):
        r"""
        Initialize the FeedForward module.

        Args:
            hidden_size (int): Input dimension.
            intermediate_size (int): Hidden dimension of the feedforward layer.
            bias (bool): If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias, **factory_kwargs)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias, **factory_kwargs)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias, **factory_kwargs)

        self.act_fn = F.silu
    
    def forward(self, x:torch.Tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class FullAttnResTransformerBlock(nn.Module):
    """Transformer block with Full Attention Residuals.
    
    Replaces standard residual connections with Attention Residuals that
    attend over all preceding layer outputs (including embedding).
    """

    def __init__(self, 
        hidden_dim: int,
        intermediate_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        bias: bool = True,
        norm_eps: float = 1e-5,
        device = None,
        dtype = None
    ):           
        super().__init__()
        self.attn_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads, attn_dropout, 
                                        resid_dropout, bias=bias, device=device, dtype=dtype)
        self.attn_attn_res = AttnRes(hidden_dim, norm_eps)

        self.ffn_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)
        self.ffn = Feedforward(hidden_dim, intermediate_dim, bias, device=device, dtype=dtype)
        self.ffn_attn_res = AttnRes(hidden_dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        layer_outputs: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass with Full Attention Residuals.
        
        Args:
            x: Input tensor [batch, seq_len, dim].
            layer_outputs: List of all preceding layer outputs (including embedding).
            attention_mask: Optional attention mask.
            
        Returns:
            A tuple of:
                - Updated hidden states [batch, seq_len, dim].
                - Updated list of intermediate layer outputs.
        """
        
        # self-attention layer
        attn_out = self.attn(self.attn_norm(x), attention_mask)
        # Apply Attention Residual
        h = self.attn_attn_res(layer_outputs, attn_out)
        layer_outputs.append(attn_out)

        # feed-forward layer
        ffn_out = self.ffn(self.ffn_norm(h))
        # Apply Attention Residual
        h = self.ffn_attn_res(layer_outputs, ffn_out)
        layer_outputs.append(ffn_out)
        
        return h, layer_outputs


class BlockAttnResTransformerBlock(nn.Module):
    """Transformer block with Block Attention Residuals.
    
    Replaces standard residual connections with Block Attention Residuals that
    attend over block-level representations.
    """

    def __init__(self, 
        hidden_dim: int,
        intermediate_dim: int,
        num_heads: int,
        block_size: int,
        layer_number: int,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        bias: bool = True,
        norm_eps: float = 1e-5,
        device = None,
        dtype = None
    ):
        super().__init__()
        self.layer_number = layer_number
        self.block_size = block_size
        
        self.attn_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads, attn_dropout, 
                                        resid_dropout, bias=bias, device=device, dtype=dtype)
        self.attn_block_attn_res = AttnRes(hidden_dim, norm_eps)

        self.ffn_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)
        self.ffn = Feedforward(hidden_dim, intermediate_dim, bias, device=device, dtype=dtype)
        self.ffn_block_attn_res = AttnRes(hidden_dim, norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_outputs: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass with Block Attention Residuals.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, dim].
            block_outputs: List of cached block-level outputs.
            attention_mask: Optional attention mask.
            
        Returns:
            A tuple of:
                - Updated partial block states [batch, seq_len, dim].
                - Updated list of block-level outputs.
        """
        partial_block = hidden_states
        # Apply Block Attention Residual before attention
        h = self.attn_block_attn_res(block_outputs, partial_block)

        if self.layer_number % (self.block_size // 2) == 0:
            block_outputs.append(partial_block)
            partial_block = None
    
        # self-attention layer
        attn_out = self.attn(self.attn_norm(h), attention_mask)
        partial_block = attn_out if partial_block is None else partial_block + attn_out
        
        # Apply Block Attention Residual before feed-forward
        h = self.ffn_block_attn_res(block_outputs, partial_block)
        # Standard feed-forward computation
        ffn_out = self.ffn(self.ffn_norm(h))
        partial_block = partial_block + ffn_out
        
        return partial_block, block_outputs


