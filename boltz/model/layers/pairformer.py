from typing import Optional, List

import torch
from torch import Tensor, nn

from boltz.data import const
from boltz.model.layers.attention import AttentionPairBias
from boltz.model.layers.attentionv2 import AttentionPairBias as AttentionPairBiasV2
from boltz.model.layers.dropout import get_dropout_mask
from boltz.model.layers.transition import Transition
from boltz.model.layers.triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from boltz.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


class PairformerLayer(nn.Module):
    """Pairformer module."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        v2: bool = False,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.dropout = dropout
        self.num_heads = num_heads
        self.post_layer_norm = post_layer_norm

        self.pre_norm_s = nn.LayerNorm(token_s)
        if v2:
            self.attention = AttentionPairBiasV2(token_s, token_z, num_heads)
        else:
            self.attention = AttentionPairBias(token_s, token_z, num_heads)

        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)

        self.tri_att_start = TriangleAttentionStartingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )

        self.transition_s = Transition(token_s, token_s * 4)
        self.transition_z = Transition(token_z, token_z * 4)

        self.s_post_norm = (
            nn.LayerNorm(token_s) if self.post_layer_norm else nn.Identity()
        )

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: Optional[int] = None,
        use_kernels: bool = False,
        use_cuequiv_mul: bool = False,
        use_cuequiv_attn: bool = False,
    ) -> tuple[Tensor, Tensor]:
        # Compute pairwise stack
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(
            z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels
        )

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(
            z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels
        )

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn or use_kernels,
        )

        dropout = get_dropout_mask(self.dropout, z, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn or use_kernels,
        )

        z = z + self.transition_z(z)

        # Compute sequence stack
        with torch.autocast("cuda", enabled=False):
            s_normed = self.pre_norm_s(s.float())
            s = s.float() + self.attention(
                s=s_normed, z=z.float(), mask=mask.float(), k_in=s_normed
            )
            s = s + self.transition_s(s)
            s = self.s_post_norm(s)

        return s, z


class PairformerModule(nn.Module):

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_blocks: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        v2: bool = False,
        *,
        scale_uniform_beta: float = 0.0,
        # Laplacian scaling
        scale_laplacian_beta: float = 0.0,
        scale_laplacian_strategy: str = "pck",
        scale_laplacian_index: int = 1,
        scale_laplacian_weights: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # store hyper‑parameters
        self.token_z = token_z
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.num_heads = num_heads
        self.post_layer_norm = post_layer_norm
        self.activation_checkpointing = activation_checkpointing

        # uniform scaling params
        self.scale_uniform_beta = scale_uniform_beta

        # Laplacian scaling params
        self.scale_laplacian_beta = scale_laplacian_beta
        self.scale_laplacian_strategy = scale_laplacian_strategy
        self.scale_laplacian_index = scale_laplacian_index
        self.scale_laplacian_weights = scale_laplacian_weights

        # transformer blocks
        self.layers = nn.ModuleList(
            [
                PairformerLayer(
                    token_s,
                    token_z,
                    num_heads,
                    dropout,
                    pairwise_head_width,
                    pairwise_num_heads,
                    post_layer_norm,
                    v2,
                )
                for _ in range(num_blocks)
            ]
        )

    def _compute_sign_matrix(self, v: Tensor) -> Tensor:
        sign_vec = v.sign()
        return sign_vec.unsqueeze(-1) * sign_vec.unsqueeze(-2)

    def _scale_uniform(self, z: Tensor) -> Tensor:
        """Apply uniform scaling to *z* if enabled and return the scaled tensor."""
        if self.scale_uniform_beta == 0.0:
            return z
        return z * (1.0 + self.scale_uniform_beta)

    def _scale_laplacian(self, z: Tensor) -> Tensor:
        """Apply Laplacian scaling (graph partition heuristic) if enabled."""
        if self.scale_laplacian_beta == 0.0:
            return z

        with torch.no_grad():
            # build graph Laplacian from pairwise norms
            A = z.norm(dim=-1)
            D = torch.diag_embed(A.sum(-1))
            L = D - A
            eps = 1e-5 * torch.eye(L.size(-1), device=L.device, dtype=L.dtype)
            eigvecs = torch.linalg.eigh(L + eps)[1]  # ascending order

            if self.scale_laplacian_strategy == "pck":
                # index 1 → Fiedler vector (second smallest eigenvalue)
                k = max(1, min(self.scale_laplacian_index, eigvecs.size(-1) - 1))
                u = eigvecs[..., k]
            elif self.scale_laplacian_strategy == "mix":
                w_raw = torch.tensor(
                    self.scale_laplacian_weights or [0.7, 0.2, 0.1],
                    device=L.device,
                    dtype=L.dtype,
                )
                w = w_raw[:3] / w_raw[:3].sum()
                u = (
                    w[0] * eigvecs[..., 1]
                    + w[1] * eigvecs[..., 2]
                    + w[2] * eigvecs[..., 3]
                )
                u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            else:
                raise ValueError(
                    f"Unsupported scale_laplacian_strategy: {self.scale_laplacian_strategy}"
                )

            S_graph = self._compute_sign_matrix(u)

        z = z * (1.0 + self.scale_laplacian_beta * S_graph).unsqueeze(-1)
        return z

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        *,
        use_kernels: bool = False,
    ) -> tuple[Tensor, Tensor]:

        # decide chunk size heuristics (kept identical to original impl.)
        if not self.training:
            chunk_size_tri_attn = (
                128 if z.shape[1] > const.chunk_size_threshold else 512
            )
        else:
            chunk_size_tri_attn = None

        # scaling pairwises features
        if self.scale_uniform_beta != 0.0:
            z = self._scale_uniform(z)
        if self.scale_laplacian_beta != 0.0:
            z = self._scale_laplacian(z)

        # transformer stack
        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                s, z = torch.utils.checkpoint.checkpoint(
                    layer,
                    s,
                    z,
                    mask,
                    pair_mask,
                    chunk_size_tri_attn,
                    use_kernels=use_kernels,
                )
            else:
                s, z = layer(
                    s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels=use_kernels
                )

        return s, z


class PairformerNoSeqLayer(nn.Module):
    """Pairformer module without sequence track."""

    def __init__(
        self,
        token_z: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.dropout = dropout
        self.post_layer_norm = post_layer_norm

        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)

        self.tri_att_start = TriangleAttentionStartingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )

        self.transition_z = Transition(token_z, token_z * 4)

    def forward(
        self,
        z: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: Optional[int] = None,
        use_kernels: bool = False,
        use_cuequiv_mul: bool = False,
        use_cuequiv_attn: bool = False,
    ) -> Tensor:
        # Compute pairwise stack
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(
            z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels
        )

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(
            z, mask=pair_mask, use_kernels=use_cuequiv_mul or use_kernels
        )

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn or use_kernels,
        )

        dropout = get_dropout_mask(self.dropout, z, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn or use_kernels,
        )

        z = z + self.transition_z(z)
        return z


class PairformerNoSeqModule(nn.Module):
    """Pairformer module without sequence track."""

    def __init__(
        self,
        token_z: int,
        num_blocks: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.post_layer_norm = post_layer_norm
        self.activation_checkpointing = activation_checkpointing

        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            self.layers.append(
                PairformerNoSeqLayer(
                    token_z,
                    dropout,
                    pairwise_head_width,
                    pairwise_num_heads,
                    post_layer_norm,
                ),
            )

    def forward(
        self,
        z: Tensor,
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> Tensor:
        if not self.training:
            if z.shape[1] > const.chunk_size_threshold:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                z = torch.utils.checkpoint.checkpoint(
                    layer,
                    z,
                    pair_mask,
                    chunk_size_tri_attn,
                    use_kernels=use_kernels,
                )
            else:
                z = layer(
                    z,
                    pair_mask,
                    chunk_size_tri_attn,
                    use_kernels=use_kernels,
                )
        return z
