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

`--cache-layers 5` implements the sliding `L1-L5 -> evict L1 -> load L6` behavior.

## CLI

Available commands:

- `show-config`
- `info`
- `inspect` (human-friendly experiment summary)
- `probe-data`
- `init-experts`
- `validate-layerwise`
- `train`
- `plot`

Important experiment knobs are first-class flags, so normal runs do not require editing JSON or using dotted overrides:

```text
--save-every N
--plot-every N
--max-steps N
--resume latest|none|PATH
--run-name NAME
--drive-root PATH
--output-dir PATH
--work-dir PATH
--cache-layers N
--superbatch N
--microbatch N
--seq-len N
--dataset NAME
--dataset-config NAME
--lr VALUE
--expert-lr VALUE
--expert-grad-dtype int8|bf16|fp32
--expert-compute-dtype bf16|fp16|fp32
--warmup-tokens N
--imbalance-threshold VALUE
--block-fraction VALUE
--threshold / --no-threshold
--snapshot-experts / --no-snapshot-experts
```

Every remaining config field is still reachable with repeatable `--set dotted.key=value`.

## Colab / Drive persistence

The Colab notebook mounts Google Drive. Using:

```text
--drive-root /content/drive/MyDrive/qwen38-threshold-moe
```

sets both:

```text
output_dir = <drive-root>/runs
work_dir   = <drive-root>/work
```

This matters because the routed expert masters are mmap-backed and updated in place. Saving only the small checkpoint while leaving `work_dir` on ephemeral Colab storage is not enough for a true resume after the runtime disappears. Drive-backed `work_dir` keeps the expert masters and Adafactor state persistent too.

For the full 500-expert/80-layer model the BF16 routed expert masters are about 117 GiB, so Drive capacity and FUSE performance must be considered before a full persistent Colab run.

## Logging / recovery

The live terminal line shows only the expert signals needed for this experiment:

- current-layer expert coverage: `E active/total`
- hottest expert and its assignment share
- per-layer routing load gap
- number of experts currently blocked by the threshold curriculum
- percentage of natural Top-K assignments rerouted because of blocking
- end-of-step mean/min expert coverage, mean/max routing gap, blocked layers/experts, and overall reroute percentage

Persistent outputs:

- `metrics.csv`
- `training.png` for loss
- `routing.png` for expert imbalance/reroute/coverage
- periodic checkpoints controlled directly by `--save-every N`
- SIGINT/SIGTERM finishes the current optimizer step and saves an interruption checkpoint
- `--resume latest` resumes the newest checkpoint
- optional `--snapshot-experts` stores an exact expert snapshot with a checkpoint; this is extremely large for the full routed bank

## Dataset streaming

Built-in aliases support the K2 Horizon aggregate, public K2 repositories, TxT360, arbitrary Hugging Face datasets, local JSONL, and weighted mixture files. Tokenized documents are packed continuously into fixed next-token sequences; the whole corpus is never downloaded first.

See `K2_RESEARCH.md` for K2 links and `COLAB.md` for the Colab runner commands.

## Important validation boundary

Run `validate-layerwise` before scaling. The entire memory-saving schedule assumes recomputation produces the same `dX` and parameter gradients as ordinary autograd before a layer's parameters are updated.
