"""Acceptance: Command-R (CohereForCausalLM) bridge reproduces HF end-to-end.

The Hub's tiny Cohere checkpoints are initialised at std=0.02, which leaves every
attention pattern near-uniform. A wrong RoPE rotation then moves logits by far
less than 1e-4, so parity against those checkpoints cannot detect it. This module
builds a Command-R-shaped model whose attention is position-sensitive, so the
logit comparison only passes when the bridge rotates Q/K the way Cohere does.

The model is constructed from a CohereConfig and seeded locally, saved to disk, and
loaded through TransformerBridge.boot_transformers like every other adapter test.
The only Hub fetch is the gpt2 tokenizer, which boot_transformers needs on disk.
"""

import tempfile
from typing import Any

import pytest
import torch
from transformers import AutoTokenizer, CohereConfig, CohereForCausalLM

from transformer_lens.model_bridge import TransformerBridge

D_VOCAB = 128
SEQ_LEN = 16
PARITY_TOL = 1e-4
# Simulated drift from Llama-style pairing on this model is ~0.5; noise is ~1e-5.
WRONG_ROTATION_MIN_DRIFT = 1e-2


def _make_command_r_shaped_model() -> Any:
    """GQA, tied embeddings, parallel block, logit_scale=1/16; unit-gain weights."""
    config = CohereConfig(
        vocab_size=D_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        logit_scale=0.0625,
        use_qk_norm=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = CohereForCausalLM(config)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim != 2:
                continue
            # Unit-gain linears give O(1) attention scores, so patterns depend on position.
            std = 1.0 if "embed_tokens" in name else param.shape[1] ** -0.5
            param.copy_(torch.randn(param.shape, generator=generator) * std)
    return model.eval()


@pytest.fixture(scope="module")
def hf_reference() -> Any:
    return _make_command_r_shaped_model()


@pytest.fixture(scope="module")
def bridge(hf_reference: Any) -> TransformerBridge:
    # Round-trip through disk so the bridge gets its own copy via the normal load
    # path; hf_reference stays an untouched, independent HF model.
    with tempfile.TemporaryDirectory() as tmpdir:
        hf_reference.save_pretrained(tmpdir)
        AutoTokenizer.from_pretrained("gpt2").save_pretrained(tmpdir)
        return TransformerBridge.boot_transformers(tmpdir, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def tokens() -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    return torch.randint(3, D_VOCAB, (2, SEQ_LEN), generator=generator)


def _max_logit_diff(bridge: TransformerBridge, hf_reference: Any, tokens: torch.Tensor) -> float:
    with torch.no_grad():
        bridge_logits = bridge(tokens, return_type="logits")
        hf_logits = hf_reference(tokens).logits
    return (bridge_logits - hf_logits).abs().max().item()


def test_adapter_requests_adjacent_pair_rotation(bridge: TransformerBridge) -> None:
    assert bridge.cfg.rotary_adjacent_pairs is True


def test_logit_scale_propagates_from_hf(bridge: TransformerBridge, hf_reference: Any) -> None:
    assert getattr(bridge.cfg, "logit_scale") == hf_reference.config.logit_scale


def test_logits_match_hf_on_position_sensitive_model(
    bridge: TransformerBridge, hf_reference: Any, tokens: torch.Tensor
) -> None:
    max_diff = _max_logit_diff(bridge, hf_reference, tokens)
    assert max_diff < PARITY_TOL, (
        f"Command-R bridge vs HF logit drift={max_diff:.2e} over {SEQ_LEN} positions. "
        "Check that Q/K are rotated in adjacent pairs (cfg.rotary_adjacent_pairs)."
    )


def test_single_position_matches_hf(bridge: TransformerBridge, hf_reference: Any) -> None:
    """Position 0 has sin=0, so it isolates everything except the rotation."""
    first = torch.tensor([[7]])
    max_diff = _max_logit_diff(bridge, hf_reference, first)
    assert max_diff < PARITY_TOL, f"Non-RoPE drift at position 0: {max_diff:.2e}"


def test_parity_check_detects_llama_style_rotation(
    bridge: TransformerBridge,
    hf_reference: Any,
    tokens: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the guard: this model must expose the wrong rotation, unlike tiny Hub models."""
    monkeypatch.setattr(bridge.cfg, "rotary_adjacent_pairs", False)
    for block in bridge.blocks:
        monkeypatch.setattr(block.attn.config, "rotary_adjacent_pairs", False)
    max_diff = _max_logit_diff(bridge, hf_reference, tokens)
    assert max_diff > WRONG_ROTATION_MIN_DRIFT, (
        f"Llama-style rotation only moved logits by {max_diff:.2e}; the fixture is "
        "too degenerate to catch a RoPE-convention regression."
    )
