"""Attention Residuals (AttnRes) modules.

Implements the core contribution of "Attention Residuals" (Kimi Team, arXiv:2603.15031):
replacing fixed residual accumulation with softmax attention over preceding layer/block
outputs via a learned pseudo-query per layer/block.
"""

import torch
import torch.nn as nn
from typing import List


class AttnRes(nn.Module):
    """Attention Residual aggregation over a sequence of preceding states.

    Given a list of previously produced states and the current state, this
    module computes a softmax-weighted sum over the full set of candidates.

    Formally:
        h = sum_i alpha_i * v_i

    where:
        v_i are the candidate states,
        alpha_i = softmax_i(q . norm(v_i)),
        q is a learned pseudo-query vector.
    """

    def __init__(
        self,
        hidden_dim: int,
        norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, 1, bias=False, **factory_kwargs)
        self.norm = nn.RMSNorm(hidden_dim, eps=norm_eps, device=device, dtype=dtype)

        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, layer_outputs: List[torch.Tensor], current_output: torch.Tensor) -> torch.Tensor:
        """Compute an attention-weighted aggregation over candidate states.

        Args:
            layer_outputs: List of previously produced states, each [batch, seq_len, dim].
            current_output: Current state [batch, seq_len, dim].

        Returns:
            Aggregated hidden state [batch, seq_len, dim].
        """
        # V: [num_states, batch, seq_len, dim]
        V = torch.stack(layer_outputs + [current_output], dim=0)
        # K: normalized states used to compute attention logits
        K = self.norm(V)
        # logits: [num_states, batch, seq_len] from the learned pseudo-query
        logits = torch.einsum("d, n b t d -> n b t", self.proj.weight.squeeze(0), K)
        # alpha: [num_states, batch, seq_len] - attention weights over states
        alpha = logits.softmax(dim=0)
        # h: [batch, seq_len, dim] - weighted sum over candidate states
        h = torch.einsum("n b t, n b t d -> b t d", alpha, V)
        return h
