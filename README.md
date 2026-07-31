# My Works

I build experimental AI systems end to end: model architectures, data and training pipelines, evaluation harnesses, retrieval/agent workflows, and product interfaces.

This repository is an honest, evidence-based map of **39 source folders** audited on **2026-07-31**, excluding this catalog repository itself. It records what each project was trying to answer, what its artifacts actually establish, what I learned, and why the work is current, paused, superseded, or archived.

It is intentionally documentation—not a bulk source dump. Private datasets, model weights, generated logs, dependency folders, and credentials remain outside GitHub.

## Start here

- [PROJECTS.md](PROJECTS.md) — all 39 folders, reviewed one by one
- [METHODOLOGY.md](METHODOLOGY.md) — scope, artifact classifications, status definitions, and claim boundaries

## Selected projects

| Project | Observed outcome | Current state |
|---|---|---|
| [**AI Research Lab / Token MOD + ATE**](PROJECTS.md#ai-research-lab) | Active research on expanding pretrained Pythia models. Shared MOD variants reported best perplexity around 3.97 versus 3.52 for a full fine-tune in one comparison. On the locked 75k setup, a pretrained Pythia-2.8B full fine-tune reached 3.2402 and fresh ATE h1/l1 reached 3.4602; those runs differ in size and steps. | **Current.** Tests, sealed data, baselines, evidence rules, and recent stage-aware experiments are present; retention and generation comparisons remain open. |
| [**FusionFormer**](PROJECTS.md#fusionformer) | The best-documented retained comparison: 28.22% LM accuracy and 95.20 perplexity versus 29.13% and 86.67 for a plain GPT. The models were not parameter matched (32.42M versus 22.74M), and the metrics survive in authored reports rather than a raw result artifact. | **Measured negative result.** Active gates did not produce an advantage in this comparison. |
| [**AB Former**](PROJECTS.md#ab-former) | A capitalization-factorization pipeline with tokenizer analysis, staged training, profiling, and smoke summaries. Baseline combined-word accuracy was 0.412 versus 0.216 and 0.137 for two structured runs; structured capitalization accuracy reached 0.733. | **Quantified smoke test, hypothesis still open.** The tiny retained corpus had no casing-driven vocabulary reduction and could not test the central efficiency claim. |
| [**NIFTY Options Algo Trader**](PROJECTS.md#algo-trader) | A broad options stack with broker abstraction, FastAPI/WebSockets, React monitoring, simulation, tests, and risk controls. Synthetic runs exist, while captured live-demo starts fail authentication and fall back to synthetic data. | **Substantial synthetic prototype.** Reliable live-broker operation and profitability are not established. |
| [**Agentic AI Coding Assistant**](PROJECTS.md#coding-assistant) | A packaged generate/run/refine loop with provider abstraction, browser UI, and a desktop IDE. | **Demo-ready prototype.** Direct local execution still needs a real sandbox, persistence, and hardening. |

Project names link to their audit entries. The source and result paths named there are local provenance pointers, not public artifacts yet.

## Catalog at a glance

| Classification | Folders | What that means |
|---|---:|---|
| **Measured** | 3 | A quantitative comparison or result summary assesses at least part of the experiment; it may still leave the main hypothesis untested. |
| **Executed** | 11 | Checkpoints, logs, generations, or other outputs prove a substantial path ran. |
| **Implemented** | 16 | The central architecture or product flow exists, but durable evaluation is missing. |
| **Fragment** | 6 | A partial sketch, historical branch, duplicate snapshot, or placeholder. |
| **Support** | 3 | Shared datasets, tokenizer assets, or design documents rather than standalone projects. |

These classifications describe evidence and catalog role, not ambition. A measured negative result is more valuable than an untested “breakthrough,” and a checkpoint proves execution—not quality.

## How the work evolved

These are inferred conceptual lineages from design overlap and dates, not verified repository ancestry:

```text
Anchor Model → Brainer → Dynamic-QKV / HoloBrain++ / Jarvis
Jarvis-flow → capitalization factorization → AB Former

Gem_model → GPT_model → early Fusion prototypes → FusionFormer
                                                   → Dynamic Transformer
                                                   → Inverted Transformer
                                                      └─ systems-cost findings
                                                         → pretrained Token MOD + ATE

my-trading-gui → trading_bot → Algo Trader
```

The first line explored stable concepts, editable memory, specialized pathways, and continual learning. The second began with explicit grammar/meaning separation, learned through a measured negative baseline comparison, and then pivoted toward controlled adaptation of pretrained models.

ChaosFormer, FractalFormer, and WildFormer are sibling architecture sketches from the same rapid exploration period. The RAG/Jarvis folders form a separate retrieval-and-agent thread, while the three trading folders show an application progression from an unsafe early dashboard to a much better-instrumented options platform.

## What the archive taught me

1. **Runnable is not validated.** Many folders have credible model code but no baseline, held-out evaluation, or saved result. Their real contribution is implementation knowledge.
2. **Negative results are useful.** FusionFormer’s gated dual streams did not beat its plain-GPT baseline and helped motivate more controlled work. The later pivot to pretrained adaptation followed routing, materialization, and writeback costs found in the intervening Dynamic/Inverted Transformer studies.
3. **Masking and data alignment are research-critical.** Several early trainers leak future tokens, use unshifted targets, or pass padding masks through the wrong attention interface. Those errors can make weak ideas appear strong.
4. **Objectives need a non-trivial learning signal.** Positive-only compatibility learning can raise every score; a router detached with `.item()` cannot learn from the LM loss; hard `argmax` routing blocks gradients.
5. **A checkpoint is an execution receipt.** It does not prove disentanglement, memory, reasoning, profitability, or production readiness without probes and comparisons.
6. **Small vertical slices beat sprawling designs.** CGI’s five engines, the game-world schemas, and several brain-inspired systems expanded faster than their end-to-end validation paths.
7. **Artifact discipline compounds.** Duplicate checkpoints, downloaded corpora, virtual environments, `node_modules`, logs, and generated charts made promising projects expensive to resume.
8. **Security is part of project quality.** The audit found unsafe local credential storage. No sensitive values or raw source from those folders are published here; source publication is blocked until credential rotation and a history-aware scan are verified.

## Why projects became dormant

There is no single cause:

- **Superseded:** Anchor Model, Gem, early GPT/Fusion branches, and the Jarvis evolution snapshots fed later work.
- **Stopped at the prototype boundary:** StorySite, the coding assistant, LoganBrain, and several architecture packages implemented the idea but not the production/evaluation infrastructure.
- **Invalid or weak experimental setup:** Future-token leakage, disconnected memory, non-differentiable routers, missing negative examples, or unshifted labels required redesign before more training.
- **Compute and storage pressure:** HoloBrain, Variants, local corpora, and checkpoint-heavy branches accumulated many gigabytes without matching artifact manifests.
- **External integration risk:** Trading and agent projects reached credentials, broker APIs, generated-code execution, concurrency, and deployment boundaries.
- **Still active:** `inverted_model/AI-Research-Lab` is current work and should not be described as deserted.

The detailed catalog keeps those distinctions explicit rather than assigning the same “abandoned” label to every old folder.

## Publication boundary

This repository contains analysis only. Before any individual source project is made public, it should receive its own dependency cleanup, license/provenance review, secret scan, history scrub where necessary, minimal reproducible test, and project-specific README.
