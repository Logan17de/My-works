# Architecture v1 — Qwen3.8-inspired Threshold MoE

This project freezes the architecture discussed in the research thread and separates **architectural identity** from **training-memory placement**. Changing `expert_cache_layers` does not change the model; it only changes how many routed expert banks are resident on the GPU at once.

## Locked model

| Component | v1 |
|---|---:|
| Decoder layers | 80 |
| Hidden width | 256 |
| Routed experts / layer | 500 |
| Always-active shared experts / layer | 4 |
| Routed experts active / token | 4 |
| Expert FFN | SwiGLU, `256 -> 2048 -> 256` |
| Expert params | 1,572,864 each |
| Routed expert params | 62,914,560,000 |
| Shared expert params | 503,316,480 |
| Vocabulary | 248,320, Qwen3.8-Flash-Next tokenizer |
| Embedding / LM head | untied |
| Base context | 8,192 |
| N-gram embedding | excluded from initial pretraining |
| MTP | excluded from initial pretraining |

The routed expert bank alone is ~117.19 GiB as BF16 masters. One 500-expert layer is ~1.465 GiB BF16 (750 MiB at one byte/parameter).

## Qwen3.8 attention/residual skeleton

The official Qwen3.8-Flash-Next model uses a 3:1 hybrid: three Gated DeltaNet (GDN) layers followed by one attention layer, plus a four-branch Gated Residual (GR) stream. This implementation keeps that layout and directly uses Transformers' Qwen4-Exp GDN implementation.

For the **initial 8K base-pretraining stage**, every fourth layer uses gated full causal attention. This is deliberate: Qwen's technical report says QSA is introduced during continued pretraining at long context. QSA can therefore be a later CPT stage rather than changing the initial experiment.

Dimensions are shrunk for hidden=256 and exposed through CLI. Defaults:

- full attention: 8 Q heads, 1 KV head, head dim 64, partial RoPE dim 16
- GDN: 4 key heads, 12 value heads, 64-d key/value heads, conv kernel 4
- GR: 4 residual branches, low-rank mixer rank 32
- RoPE theta: 10,000,000

## Router and threshold curriculum

The experimental default is **sigmoid scores + Top-4**. This is intentionally configurable:

```bash
--set model.router_activation=sigmoid
# control comparison with the current HF reference implementation
--set model.router_activation=softmax
```

Important correction: the current Hugging Face `Qwen4ExpTextTopKRouter` implementation uses **softmax**, not sigmoid. Sigmoid is retained here because it is part of this experiment's locked hypothesis, not because it is claimed to be identical to Qwen's released router.

Each layer records both natural Top-4 and executed Top-4 choices. Thresholding acts only on execution:

1. Train freely for `threshold.warmup_tokens=N`.
2. Measure cumulative **executed** expert assignment counts per layer.
3. Gap is `(most_used - least_used) / most_used`.
4. If gap > 30%, block the most-used 30% of experts (150/500 by default).
5. Re-run Top-K over the remaining experts. Thus a blocked natural Top-4 candidate is replaced by the next-ranked available candidate.
6. Masks change only between optimizer steps, never between forward and backward recomputation.
7. Set `threshold.enabled=false` for unrestricted routing later.

`warmup_tokens=-1` means "not supplied yet" and disables threshold activation, matching the decision to provide N at training time.

## Four shared + four routed experts

All four shared experts are always executed. Their outputs are averaged by default and added to the Top-4 routed output. This makes eight expert FFNs active per layer per token while only four are routed.

## Layer-wise training

The runtime does not retain an 80-layer autograd graph.

**Forward**

1. Run a layer under `no_grad`.
2. Store the layer boundary activation in host memory.
3. Continue to the next layer.

**Backward**

1. Start at layer 80.
2. Reload its BF16 expert master into the GPU cache if needed.
3. Recompute the layer with autograd.
4. Compute `dX` **before changing any parameters**.
5. Apply the resident-parameter optimizer and routed-expert Adafactor update.
6. Free the layer gradient and expert GPU copy.
7. Continue to layer 79.

This ordering is the critical validation boundary: a small reference experiment should verify gradients against ordinary end-to-end autograd before interpreting throughput or scaling results.

## Sliding expert-layer cache

`training.expert_cache_layers` controls a sliding window only:

```text
capacity = 5

[L1 L2 L3 L4 L5]
 execute L1
 evict L1 + load L6
[L2 L3 L4 L5 L6]
 execute L2
 evict L2 + load L7
...
```

The reverse traversal uses the same cache in the other direction. If a 96 GiB card has spare space, increase the number from CLI without changing weights, routing, optimizer semantics, or checkpoints.

## Optimizer split

- routed experts: momentum-free factored **Adafactor**, CPU state
- router / attention / GR / shared experts / embedding / LM head: **AdamW** baseline
- routed gradients: current layer only; configurable `int8`, `bf16`, or `fp32` CPU transfer
- authoritative routed expert masters: BF16 CPU RAM or memory-mapped files

The Adafactor state for these matrix shapes is roughly ~1 GiB for the full routed bank instead of two full Adam moments.

## FP8

The first implementation deliberately validates BF16 expert compute first. Direct FP8 autograd is guarded rather than silently emulated. Once layer-wise gradients are verified, a Transformer Engine FP8 shadow/cache can replace the BF16 staging function without changing the CPU masters or training schedule.
