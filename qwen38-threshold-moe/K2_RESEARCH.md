# K2 Horizon pretraining-data notes

## Exact aggregate link

The K2 Horizon model cards reference this aggregate pretraining dataset:

https://huggingface.co/datasets/IFM/K2-Horizon-Pretrain-Data

At the time this project was prepared, anonymous access returned HTTP 401. The trainer supports this dataset directly through `data.source=k2-horizon` and forwards `HF_TOKEN` when set, so no code change is required if access is available to the user.

## Public K2 Horizon training-data repositories

- TxT360-v2: https://huggingface.co/datasets/IFM/TxT360-v2
- Code-Reasoning: https://huggingface.co/datasets/IFM/Code-Reasoning
- Math-Reasoning: https://huggingface.co/datasets/IFM/Math-Reasoning
- Pretrain-Behaviors: https://huggingface.co/datasets/IFM/Pretrain-Behaviors
- SFT-Reasoning: https://huggingface.co/datasets/IFM/SFT-Reasoning
- Original TxT360: https://huggingface.co/datasets/LLM360/TxT360

The public repositories are Parquet-backed and can be streamed with Hugging Face Datasets. `IFM/TxT360-v2` exposes web/high-quality and QA subsets; the code, math, and behavior repositories expose multiple reasoning and rewrite subsets.

## What IFM reports

IFM states that K2 Horizon 3.7B, 7B, 32B, and 36B-A4B were trained on the same 22T-token sequence. Their announcement describes a mixture spanning web, code, mathematics, scientific, multilingual, domain-specific, and synthetic sources. Nearly 17% of the pretraining corpus is described as explicit problem-solving/reasoning trajectories, and approximately 10T synthetic tokens were used.

Source announcement:

https://ifm.ai/blog/k2/

## Data aliases built into this trainer

- `k2-horizon` -> `IFM/K2-Horizon-Pretrain-Data`
- `k2-txt360-v2` -> `IFM/TxT360-v2`
- `txt360` -> `LLM360/TxT360`
- `k2-code` -> `IFM/Code-Reasoning`
- `k2-math` -> `IFM/Math-Reasoning`
- `k2-behaviors` -> `IFM/Pretrain-Behaviors`
- `k2-sft` -> `IFM/SFT-Reasoning`
- `hf:ORG/DATASET` -> arbitrary Hugging Face dataset
- `local:/path/file.jsonl` -> local JSONL

`configs/k2_public_mix.example.json` is intentionally only a template. Its weights are **not** claimed to reproduce K2 Horizon's exact mixture. Replace them after verifying IFM's official mixture recipe.
