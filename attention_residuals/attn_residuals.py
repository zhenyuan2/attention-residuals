"""Attention Residuals (AttnRes) modules.

Implements the core contribution of "Attention Residuals" (Kimi Team, arXiv:2603.15031):
replacing fixed residual accumulation with softmax attention over preceding layer/block
outputs via a learned pseudo-query per layer/block.
"""

import torch
import torch.nn as nn
from typing import List


class AttnRes(nn.Module):
    """Full Attention Residuals: attends over ALL preceding layer outputs.

    For layer l, the hidden state is computed as:
        h_l = sum_i alpha_{l,i} * v_i

    where:
        v_i are the outputs of each preceding layer (including embedding),
        alpha_{l,i} = softmax_i(q_l . norm(v_i))
        q_l is a learned pseudo-query vector (d-dimensional).
    """

    def __init__(self, hidden_dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, 1, bias=False)
        self.norm = nn.RMSNorm(hidden_dim, eps=norm_eps)

        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, layer_outputs: List[torch.Tensor], current_output: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted aggregation over layer outputs.

        Args:
            layer_outputs: List of previous layer outputs, each [batch, seq_len, dim].
            current_output: Current layer output [batch, seq_len, dim].

        Returns:
            Aggregated hidden state [batch, seq_len, dim].
        """
        # V: [num_layers, batch, seq_len, dim]
        V = torch.stack(layer_outputs + [current_output], dim=0)
        # K: normalized values for computing attention logits
        K = self.norm(V)
        # logits: [num_layers, batch, seq_len] via einsum with pseudo-query
        logits = torch.einsum("d, n b t d -> n b t", self.proj.weight.squeeze(0), K)
        # alpha: [num_layers, batch, seq_len] - attention weights over depth
        alpha = logits.softmax(dim=0)
        # h: [batch, seq_len, dim] - weighted sum
        h = torch.einsum("n b t, n b t d -> b t d", alpha, V)
        return h
