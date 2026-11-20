# Special Cases

```{warning}
`HookedTransformer` is deprecated as of TransformerLens 3.0 and will be removed in the next major version. New code should use [`TransformerBridge`](migrating_to_v3.md) instead. Existing `HookedTransformer` code continues to work through the 3.x branch via a compatibility layer. See the [migration guide](migrating_to_v3.md) for conversion recipes.
```

## Mixture of Experts error rates
Due to the Top-K gating performed in the hidden layer of Mixture of Experts models, small errors can be amplified 
greatly in cases where a different expert is selected, which leads to a higher than normal variance in the error rate
of the final logits. In testing done on Mixtral running in half precision, the standard deviation of the absolute error 
rate of the logits compared to those from the default model was found to be around 2e-3.

There are two main ways to mitigate this:

1. **Skip weight preprocessing.** On the bridge, simply load with `TransformerBridge.boot_transformers(...)` and do not call `enable_compatibility_mode()` - the bridge preserves raw HF weights by default, so no additional flag is needed. On the legacy `HookedTransformer` path, use `HookedTransformer.from_pretrained_no_processing` instead of `HookedTransformer.from_pretrained`.
2. **Increase the precision of the data type used in the model.**

## Qwen3.5 text-only models

Qwen3.5 has no `HookedTransformer.from_pretrained` legacy path. Load it through `TransformerBridge`. Dense text-only checkpoints can be loaded with:

```python
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers("Qwen/Qwen3.5-0.8B")
```

Qwen3.5 uses a hybrid stack. Full-attention layers expose the usual hooks under
`blocks.N.attn.*`; linear-attention layers expose GatedDeltaNet hooks under
`blocks.N.linear_attn.*`, including `hook_q_pre_conv`, `hook_k_pre_conv`,
`hook_v_pre_conv`, `hook_beta`, `hook_log_decay`, `hook_recurrence_out`, and
`hook_out`. Full multimodal `Qwen3_5ForConditionalGeneration`, image/video
inputs, and Qwen3.5 MoE checkpoints are not supported by this adapter.

## Cohere Command R and Aya

`CohereForCausalLM` checkpoints load through `TransformerBridge` only; there is no
`HookedTransformer` path. Support depends on one config flag, `use_qk_norm`:

| Checkpoint family | `use_qk_norm` | Status |
|---|---|---|
| Command R (`c4ai-command-r-v01`, `c4ai-command-r-08-2024`) | `false` | Adapter covers this configuration. No full-size checkpoint has been run through `verify_models` yet. |
| Aya 23 8B, Aya Expanse 8B / 32B | `false` | Same as Command R. |
| Command R+ (`c4ai-command-r-plus`, `c4ai-command-r-plus-08-2024`, and their quantized variants) | `true` | **Not supported.** Loading raises a `RuntimeError`. |
| Command R7B, Command A (`Cohere2ForCausalLM`) | n/a | **Not supported.** Different architecture class with no adapter. |

Command R+ applies a per-head LayerNorm to queries and keys before the rotary
embedding. The Cohere adapter does not map that norm, and the attention bridge
refuses to run a model whose `q_norm`/`k_norm` it would otherwise skip. The check
runs when the bridge wraps the model, which is after the weights have been
downloaded and loaded, so check `use_qk_norm` in `config.json` first.

The only Cohere entry marked verified in the model table is a tiny random test
checkpoint. Its attention patterns are close to uniform, so its logit-parity score
says little about the rotary-embedding handling that real checkpoints depend on.
Treat full-size Command R and Aya checkpoints as unverified until a
`verify_models` run records scores for them.

## DeepSeek V2, V3 and R1

"DeepSeek" covers several HuggingFace architecture classes, and "R1" covers two
unrelated ones. Which adapter loads a checkpoint depends on its class, not its name.
Statuses below are from the model registry snapshot generated 2026-06-25.

| Checkpoints | HF class | Adapter | Status |
|---|---|---|---|
| DeepSeek-V2, V2-Lite, V2.5, Coder-V2 | `DeepseekV2ForCausalLM` | `deepseek_v2` | Adapter present, nothing verified. The one tiny checkpoint that was run failed. Official checkpoints were skipped, nearly all as too large for the verification machine; DeepSeek-V2-Lite has a phase-1 score recorded but no completed run. |
| DeepSeek-V3, V3-0324, V3.1, R1, R1-0528, R1-Zero | `DeepseekV3ForCausalLM` | `deepseek_v3` | Adapter present. Two tiny random checkpoints are verified (one is a GigaChat3 test model of the same class) and five failed (two of those at download). No full-size checkpoint has been run. |
| R1 distills (`DeepSeek-R1-Distill-Llama-*`, `DeepSeek-R1-Distill-Qwen-*`, `DeepSeek-R1-0528-Qwen3-8B`) | `LlamaForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM` | Llama, Qwen2, Qwen3 | Verified. The 70B Llama and 32B Qwen distills are architecture-validation only. These are standard-attention models and none of the notes below apply to them. |
| DeepSeek-V3.2, DeepSeek-V4, DeepSeek-VL2, DeepSeek-OCR-2, `deepseek-moe-16b` | `DeepseekV32ForCausalLM`, `DeepseekV4ForCausalLM`, `DeepseekVLV2ForCausalLM`, `DeepseekOCR2ForCausalLM`, `DeepseekForCausalLM` | none | **Not supported.** No adapter exists for these classes. |

What the two DeepSeek adapters expose, on the checkpoints they do load:

- Attention is Multi-Head Latent Attention (MLA). The bridge adds
  `blocks.N.attn.hook_q_latent` and `hook_kv_latent` for the compressed
  representations, alongside `hook_q`, `hook_k`, `hook_v`, `hook_rot_q`,
  `hook_rot_k`, `hook_attn_scores`, `hook_pattern` and `hook_z`.
- DeepSeek-V2-Lite has no query compression, so `hook_q_latent` does not fire there.
- MLA blocks do not have `hook_q_input`, `hook_k_input`, `hook_v_input` or
  `hook_attn_in`, with or without `enable_compatibility_mode()`.
- The MoE router is hookable on V3 (`blocks.N.mlp.gate`) but not on V2, where the
  HF model reads the router weight directly instead of calling the module.
- Neither adapter declares weight conversions, and the verified tiny V3
  checkpoints still fail the `value_bias_folding` and `weight_modification`
  checks. Treat weight processing under `enable_compatibility_mode()` as
  unverified on these models.
