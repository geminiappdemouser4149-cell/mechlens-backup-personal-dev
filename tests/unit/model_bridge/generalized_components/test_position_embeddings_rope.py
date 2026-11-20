"""Tests for the adjacent-pair RoPE helpers used by PositionEmbeddingsAttentionBridge."""

import torch
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
)

from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    _apply_rotary_pos_emb_adjacent_pairs,
    _rotate_adjacent_pairs,
)

BATCH, HEADS, SEQ, HEAD_DIM = 2, 3, 6, 8


def _interleaved_cos_sin(seq: int = SEQ, head_dim: int = HEAD_DIM) -> tuple:
    """cos/sin as CohereRotaryEmbedding builds them: each frequency repeated in place."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq).float(), inv_freq)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)
    return emb.cos()[None], emb.sin()[None]


def _random_qk() -> tuple:
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM, generator=generator)
    k = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM, generator=generator)
    return q, k


class TestRotateAdjacentPairs:
    def test_known_values(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.equal(_rotate_adjacent_pairs(x), torch.tensor([-2.0, 1.0, -4.0, 3.0]))

    def test_double_rotation_negates(self) -> None:
        q, _ = _random_qk()
        assert torch.equal(_rotate_adjacent_pairs(_rotate_adjacent_pairs(q)), -q)


class TestApplyRotaryAdjacentPairs:
    def test_matches_explicit_2d_rotation(self) -> None:
        q, k = _random_qk()
        cos, sin = _interleaved_cos_sin()
        q_rot, _ = _apply_rotary_pos_emb_adjacent_pairs(q, k, cos, sin)
        theta = torch.atan2(sin[0, :, ::2], cos[0, :, ::2])  # [seq, head_dim/2]
        even, odd = q[..., ::2], q[..., 1::2]
        expected_even = even * theta.cos() - odd * theta.sin()
        expected_odd = even * theta.sin() + odd * theta.cos()
        assert torch.allclose(q_rot[..., ::2], expected_even, atol=1e-6)
        assert torch.allclose(q_rot[..., 1::2], expected_odd, atol=1e-6)

    def test_preserves_norm(self) -> None:
        q, k = _random_qk()
        cos, sin = _interleaved_cos_sin()
        q_rot, k_rot = _apply_rotary_pos_emb_adjacent_pairs(q, k, cos, sin)
        assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
        assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-5)

    def test_scores_depend_only_on_relative_position(self) -> None:
        cos, sin = _interleaved_cos_sin()
        generator = torch.Generator().manual_seed(1)
        q_vec = torch.randn(HEAD_DIM, generator=generator)
        k_vec = torch.randn(HEAD_DIM, generator=generator)
        q = q_vec.expand(1, 1, SEQ, HEAD_DIM)
        k = k_vec.expand(1, 1, SEQ, HEAD_DIM)
        q_rot, k_rot = _apply_rotary_pos_emb_adjacent_pairs(q, k, cos, sin)
        scores = q_rot[0, 0] @ k_rot[0, 0].T  # [seq, seq]
        for offset in range(SEQ - 1):
            diagonal = torch.diagonal(scores, offset=offset)
            assert torch.allclose(diagonal, diagonal[0].expand_as(diagonal), atol=1e-5)

    def test_llama_pairing_is_not_equivalent_on_interleaved_cos_sin(self) -> None:
        """The regression this helper exists for: Llama's helper is wrong here."""
        q, k = _random_qk()
        cos, sin = _interleaved_cos_sin()
        q_rot, _ = _apply_rotary_pos_emb_adjacent_pairs(q, k, cos, sin)
        q_llama, _ = llama_apply_rotary_pos_emb(q, k, cos, sin)
        # Position 0 has sin=0, so both conventions are the identity there.
        assert torch.allclose(q_rot[:, :, 0], q_llama[:, :, 0], atol=1e-6)
        assert (q_rot[:, :, 1:] - q_llama[:, :, 1:]).abs().max().item() > 0.1
        # Llama pairing on interleaved cos/sin is not even a rotation.
        assert not torch.allclose(q_llama.norm(dim=-1), q.norm(dim=-1), atol=1e-3)

    def test_restores_input_dtype(self) -> None:
        q, k = _random_qk()
        cos, sin = _interleaved_cos_sin()
        q_rot, k_rot = _apply_rotary_pos_emb_adjacent_pairs(q.half(), k.half(), cos, sin)
        assert q_rot.dtype == torch.float16
        assert k_rot.dtype == torch.float16
