# K2 Horizon research notes used by this trainer

Checked against IFM's K2 Horizon release on 2026-09-07.

## What is useful for this project

IFM reports that K2 Horizon 3.7B, 7B, 32B and MoVA-36B-A4B were trained on **the same 22 trillion-token sequence**. The pretraining mixture includes web, code, mathematics, science, multilingual/domain data and synthetic data. IFM says nearly **17% of pretraining is explicit problem-solving/reasoning trajectories** and roughly **10T synthetic tokens** were used.

That makes K2 unusually useful for our architecture experiment: the data recipe is designed for large-scale pretraining and the dense/sparse family gives a controlled reference for later comparisons.

Sources:
- https://ifm.ai/blog/k2/
- https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B

## Public data lineage

The model cards point to:

- `IFM/K2-Horizon-Pretrain-Data`
- `IFM/K2-Horizon-Midtrain-Data`

and identify the pretraining lineage as TxT360. TxT360 globally deduplicates 99 CommonCrawl snapshots together with curated sources and exposes metadata suitable for deliberate upsampling.

Public lineage:
- https://github.com/LLM360/TxT360
- https://huggingface.co/datasets/LLM360/TxT360

The newer K2 Horizon dataset series is also appearing as separately streamable repositories. At research time these included:

- `IFM/TxT360-v2` — web and QA text; public configs include `web-high-nltk-qa`, `web-high-medium`, `txt360-qa`
- `IFM/Code-Reasoning`
- `IFM/Math-Reasoning`
- `IFM/SFT-Reasoning`
- `IFM/Pretrain-Behaviors`

The trainer therefore does **not** hard-code an invented K2 mixture. It supports the exact aggregate dataset ID when accessible and exposes arbitrary Hugging Face sources/configs through CLI.

## Access status / important limitation

During this research pass, anonymous fetching of `IFM/K2-Horizon-Pretrain-Data` returned HTTP 401 through the research crawler even though the model cards reference it. The larger K2 model card also describes data/training-code publication as a rolling release. Therefore this repo does **not** claim that it has reconstructed the exact 22T-token mixture weights.

Behavior is explicit:

```bash
# Exact K2 aggregate ID. Set HF_TOKEN if the dataset requires authentication.
python data_probe.py --config configs/full_g4.json

# Public K2/TxT360 component for pipeline testing now.
python data_probe.py --config configs/colab_smoke.json \
  --set data.source=k2-txt360-v2 \
  --set data.config_name=web-high-nltk-qa

# Any HF dataset/config without code changes.
python data_probe.py --config configs/colab_smoke.json \
  --set data.source=hf:ORG/DATASET \
  --set data.config_name=CONFIG
```

## Training ideas taken from K2 / TxT360 lineage

Useful ideas, without pretending they are the exact Horizon-36B recipe:

- 8K is a real base-pretraining context in K2's released stage lineage.
- Keep natural text as the grounding distribution, then deliberately introduce reasoning/synthetic domains.
- Deduplicate first; make repetition/upsampling an explicit quality decision rather than an accident.
- Keep data source metadata so later expert-specialization analysis can correlate routes with domains.
- Stream rather than materialize the corpus locally.

The previous K2-V2 report is especially useful for public methodology: it documents TxT360 data curation, code/math sources, and quality/duplicate-based upsampling. We use those as research references, not as a claim about exact K2 Horizon mixture weights.

## xLLM provenance

K2 model cards reference `LLM360/xllm` and stage-specific commits for training provenance. The referenced 36B commit could not be independently fetched through the connected GitHub API during this pass, so no unverified implementation detail from that commit was copied into this repository. The project instead uses publicly inspectable Qwen/Transformers code for the model skeleton and the K2/TxT360 releases for the data path.
