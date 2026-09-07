# Qwen3.8-inspired Threshold MoE pretraining experiment

Single-GPU research implementation for the 80-layer, hidden-256, 500-routed-expert architecture discussed in the design thread.

## Current architecture

- 80 layers, hidden 256
- 500 routed experts/layer
- four shared experts/layer
- softmax + Top-4 router only
- threshold-routing curriculum after configurable warmup
- Qwen-style 3 Gated DeltaNet : 1 gated full-attention layout
- four-branch Gated Residual
- Qwen partial RoPE (`theta=1e7`) in full-attention layers
- expert SwiGLU `256 -> 2048 -> 256`
- Qwen3.8 tokenizer, 248,320 vocabulary, untied embedding/head
- initial 8K pretraining; no n-gram embedding or MTP

## Memory strategy

Routed expert masters live in BF16 CPU mmap/RAM. A configurable sliding window of expert layers is copied to GPU. Forward stores only layer-boundary activations on CPU. Backward recomputes one layer at a time, computes `dX`, transfers the routed expert gradient to CPU (INT8 by default), updates routed masters with factored Adafactor, updates resident parameters with AdamW, and frees the graph.

`training.expert_cache_layers=5` implements the sliding `L1-L5 -> evict L1 -> load L6` behavior.

## CLI

Every config value can be changed without editing code. Available commands:

- `show-config`
- `info`
- `probe-data`
- `init-experts`
- `validate-layerwise`
- `train`
- `plot`

Use repeatable `--set dotted.key=value` arguments for CLI-only changes.

## Logging / recovery

The live terminal line now shows only the expert signals needed for this experiment rather than dumping every expert:

- current layer expert coverage: `E active/total`
- hottest expert and its assignment share
- per-layer routing load gap
- number of experts currently blocked by the threshold curriculum
- percentage of natural Top-K assignments actually rerouted because of blocking
- at the end of each optimizer step: mean/min expert coverage across layers, mean/max routing gap, number of blocked layers/experts, and overall reroute percentage

Persistent outputs:

- `metrics.csv` with the same experiment-level routing summaries
- `training.png` for loss
- `routing.png` for mean/max routing gap, reroute rate, and mean active experts
- periodic checkpoints every `training.save_every_steps`
- SIGINT/SIGTERM requests finish the current optimizer step and then save an interruption checkpoint
- `training.resume=latest` resumes the latest checkpoint
- routed expert files are mmap-backed and therefore updated in place
- optional `training.snapshot_experts=true` stores an exact expert snapshot with a checkpoint; this can be very large for the 63B routed bank

## Dataset streaming

Built-in aliases support the K2 Horizon aggregate, public K2 repositories, TxT360, arbitrary Hugging Face datasets, local JSONL, and weighted mixture files. Tokenized documents are packed continuously into fixed next-token sequences; the whole corpus is never downloaded first.

See `K2_RESEARCH.md` for K2 links and `COLAB.md` for a command-only Colab flow.

## Important validation boundary

Run `validate-layerwise` before scaling. The entire memory-saving schedule assumes recomputation produces the same `dX` and parameter gradients as ordinary autograd before a layer's parameters are updated.
