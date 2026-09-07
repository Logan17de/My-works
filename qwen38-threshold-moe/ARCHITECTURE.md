# Architecture v1 — Qwen3.8-inspired Threshold MoE

This is the locked initial-pretraining architecture. Runtime memory placement (for example `expert_cache_layers`) does not change model identity.

## Locked model

| Component | v1 |
|---|---:|
| Decoder layers | 80 |
| Hidden width | 256 |
| Routed experts / layer | 500 |
| Always-active shared experts / layer | 4 |
| Routed experts active / token | Top-4 |
| Expert FFN | SwiGLU `256 -> 2048 -> 256` |
| Routed expert params | 62,914,560,000 |
| Shared expert params | 503,316,480 |
| Vocabulary | 248,320, Qwen3.8-Flash-Next tokenizer |
| Embedding / LM head | untied |
| Initial context | 8,192 |
| N-gram embedding | excluded initially |
| MTP | excluded initially |

The routed bank is about 117.2 GiB as BF16 masters. One 500-expert layer is about 1.465 GiB BF16.

## Qwen attention / position design

The initial 8K stage keeps Qwen3.8's 3:1 hybrid structure:

- three Qwen Gated DeltaNet layers
- one gated full causal-attention layer
- four-branch Gated Residual stream around attention and MoE sublayers

Qwen Sparse Attention is a later continued-pretraining option; it is not required for the first 8K experiment.

Position handling follows Qwen:

- **partial RoPE**, not learned absolute positional embeddings
- RoPE is applied to Q/K in the full-attention layers
- `rope_theta = 10,000,000`
- attention head dim = 64 in this scaled model
- partial rotary factor = 0.25, therefore 16 rotary dimensions/head
- Gated DeltaNet uses its recurrent state rather than an absolute positional table

Scaled head defaults:

- full attention: 8 Q heads, 1 KV head, head dim 64
- GDN: 4 key heads, 12 value heads, key/value head dim 64
- GDN conv kernel: 4
- Gated Residual: 4 branches, low-rank mixer rank 32

## Router

The routed-expert router is **softmax only**, matching the current Qwen4-Exp / Qwen3.8 reference behavior:

1. linear projection `hidden -> 500 logits`
2. FP32 softmax across experts
3. Top-4 selection
4. selected probabilities are normalized to sum to 1

There is no sigmoid-router option in this project. Sigmoid gates still exist inside Qwen's attention/GDN/Gated-Residual mechanisms; those are not MoE routers.

## Threshold curriculum

Thresholding changes executed expert choices only during pretraining:

1. train freely for `threshold.warmup_tokens`
2. accumulate executed expert counts per layer
3. imbalance gap = `(most_used - least_used) / most_used`
4. when the gap exceeds 30%, block the hottest 30% of experts (150 of 500)
5. Top-K continues down the same softmax ranking to choose the next available experts
6. both natural Top-4 and executed Top-4 counts are recorded
7. masks only change between optimizer steps, so forward and backward recomputation use the same mask
8. set `threshold.enabled=false` to remove the curriculum later

## Shared experts

All four shared experts execute for every token. Their outputs are averaged and added to the weighted Top-4 routed expert output. Eight FFNs therefore execute per token/layer: four shared + four routed.

## Layer-wise exact-backprop schedule

The trainer deliberately avoids an 80-layer autograd graph.

Forward:

1. embed tokens and create the four residual streams
2. load/cache routed expert banks for the current window
3. execute one layer under `no_grad`
4. save the layer-boundary activation to CPU
5. move to the next layer

Backward:

1. recompute LM-head/final-mixer work and obtain the gradient of the final boundary
2. traverse layer 80 -> layer 1
3. reload or reuse that layer's routed expert bank
4. recompute the layer with autograd
5. compute `dX` before any parameter update
6. transfer routed-expert gradients to CPU (`int8`, `bf16`, or `fp32`)
7. update BF16 CPU expert masters with factored Adafactor
8. update attention/GDN/router/shared/GR parameters with AdamW
9. release the layer graph and continue backward
10. finally recompute/update the token embedding

`validate-layerwise` compares direct autograd gradients with recomputed layer-wise gradients on a tiny model before a large run.

## Sliding expert cache

`training.expert_cache_layers=N` controls how many routed-expert layers occupy VRAM at once.

For `N=5`:

```text
[L1 L2 L3 L4 L5]
execute L1 -> evict L1 -> load L6
[L2 L3 L4 L5 L6]
execute L2 -> evict L2 -> load L7
...
```

Backward uses the same window in reverse. Increasing this value uses spare VRAM to reduce reload stalls without changing the model.

## Expert storage and optimizer

Routed experts:

- authoritative master: BF16 CPU mmap/RAM
- GPU copy: current sliding cache
- gradient transfer: INT8 by default, configurable to BF16/FP32
- optimizer: momentum-free factored Adafactor on CPU

Resident parameters:

- token embedding / LM head
- GDN and full-attention weights
- Gated Residual parameters
- routers
- four shared experts/layer

These remain on GPU and use AdamW.

## FP8

The Colab/A100 correctness path uses BF16. A100 does not provide native FP8 Tensor Core training. The architecture keeps the routed-expert cache boundary isolated so a Transformer Engine FP8 cache backend can later replace BF16 staging on Blackwell without changing CPU masters, routing, data, checkpoints, or layer-wise scheduling.
