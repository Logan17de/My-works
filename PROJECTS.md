# Project-by-project audit

This catalog covers 39 sibling source folders found under the local archive root on 2026-07-31, excluding this `My-works` repository. Folder names are preserved so each entry can be traced back to its local evidence.

The labels **Measured**, **Executed**, **Implemented**, **Fragment**, and **Support** use the definitions in [METHODOLOGY.md](METHODOLOGY.md). “Why it paused” is an evidence-based inference, not a claim about the author’s private intent. “Key evidence” names audited local provenance; those paths are not public links.

<a id="ab-former"></a>

## AB_Former — AB Former

`Research experiment` · `Measured` · `Paused after smoke study` · Last substantive evidence: `2026-05-05`

**Started to ask:** Can capitalization be factored out of surface tokens into lowercase/base tokens plus character-level casing bits, reducing vocabulary duplication and improving data efficiency?

**What the artifacts establish:** A complete PyTorch experiment exists: tokenizer normalization, staged model growth, confidence-gated capitalization loss, evaluation, profiling, checkpoints, and three Stage-1 summaries. Normalization merged 52,696 duplicate regular tokens in the exported tokenizer. In the tiny retained smoke runs, however, baseline combined-word accuracy was 0.412, versus 0.216 and 0.137 for two structured runs; the structured model’s capitalization accuracy reached 0.733.

**Lessons:** The pipeline works, but the smoke corpus had zero casing-driven vocabulary reduction and only 1,042 total train/validation tokens, so it could not test the main efficiency hypothesis. The structured path also added parameters and ran more slowly. A larger casing-rich corpus, matched parameter budgets, later stages, and multiple seeds are required.

**Why it paused:** The evidence ends after smoke validation, before Stage 2/3 or a full-scale comparison. Missing README, dependency metadata, and automated tests make continuation harder, but this is better described as an unfinished measured study than a deserted sketch.

**Key evidence:** `model.py`, `dataset.py`, `train.py`, `eval.py`, `profiler.py`, `Tokenizer/`, `runs/*/summary.json`

## Advanced-RAG Model — Fusion Memory Model

`Research prototype` · `Executed` · `Paused at integration stage` · Last substantive evidence: `2025-08-12`

**Started to ask:** Can a small language model combine style controls, learned meta-routing, and editable FAISS-backed memory?

**What the artifacts establish:** Tone/type/mood controls, a `MetaPredictionQKV` module, a dual-index memory concept, an Ollama-generated 188-example dataset, trainer, chat path, and an epoch-0 checkpoint were assembled. The checkpoint proves an initial run, not a functioning RAG system.

**Lessons:** The default meta-QKV path is disabled and is not trained by the retained trainer; the memory embedding is disconnected and no memories or retrieval evaluation remain. The Transformer has no causal mask and trains with unshifted labels, allowing identity copying. “RAG” therefore describes the intended direction more than the implemented behavior.

**Why it paused:** Only epoch 0 survives from a six-epoch default, with no metrics, tests, requirements, populated memory, or evaluation. The concentrated one-day implementation likely stopped when the integration gaps became the next required work.

**Key evidence:** `config.py`, `model.py`, `meta_qkv.py`, `memory_system.py`, `trainer.py`, `chat.py`, `data.jsonl`, `checkpoints/epoch_0.pt`

<a id="coding-assistant"></a>

## agentic_project_app — Agentic AI Coding Assistant

`Application prototype` · `Implemented` · `Paused at packaged demo` · Last substantive source evidence: `2025-08-23`

**Started to ask:** Can a provider-independent, Windsurf-like assistant generate code, run it locally, capture failures, and ask an LLM to repair the result?

**What the artifacts establish:** The generate/run/refine loop, OpenAI/Gemini/stub provider abstraction, FastAPI endpoints, browser controls, Tkinter desktop IDE, and a PyInstaller executable were built. The stub provider is a useful offline seam for testing the workflow.

**Lessons:** Directly executing generated code in local subprocesses is the decisive safety boundary. Persistent memory, project indexing, sandboxing, non-blocking LLM calls, and automated tests are still absent. The 416 MB desktop binary also shows the distribution cost of packaging the stack as-is.

**Why it paused:** The retained work reaches a convincing packaged proof-of-concept but not a production-safe agent. No durable test or usage results show a later hardening phase.

**Key evidence:** `README.md`, `agentic_code_assistant.py`, `backend/app.py`, `backend/llm_clients.py`, `desktop_app.py`, `frontend/`, `dist/desktop_app.exe`

<a id="algo-trader"></a>

## Algo Trader — NIFTY Options Algo Trader

`Trading application` · `Executed` · `Paused pending live-broker validation` · Last substantive evidence: `2026-04-08`

**Started to ask:** Can ATM CALL/PUT entries be automated around high-confidence breakout/breakdown signals while combining spot/futures volume, price, open interest, wave stops, and explicit risk controls?

**What the artifacts establish:** A substantial Python/React system exists with FastAPI/WebSocket control, React monitoring, broker abstraction, live and synthetic feeds, demo/real modes, backtesting, authentication checks, option sizing, tests, and deployment packaging. Recorded demo runs and CSVs prove simulation paths ran.

**Lessons:** Retained logs show Groww authentication failing with HTTP 400 and demo execution falling back to synthetic data, so live order placement is not established. One April simulation file records ten repeated-run losses totaling roughly 94.9k, while a November file contains one win and one loss with a small positive net; neither is a profitability study. Strategy tuning, run identity, broker contract tests, and reproducible full-suite results matter more than feature count.

**Why it paused:** The application is a substantial synthetic prototype, but broker endpoints still carry adjustment notes, full dependency metadata is incomplete, and observed fallback behavior differs from the README’s implied live path. Credential and integration work are the most visible blockers.

**Publication boundary:** Unsafe local credential storage was detected. Sensitive material is excluded; source release requires verified rotation and a history-aware scan.

**Key evidence:** `README.md`, `strategy.py`, `trend_analyzer.py`, `api.py`, `broker_groww.py`, `main_live.py`, `tests/`, `logs/`, `algo-dashboard/`

## Anchor Model — Anchoring AI Project

`Historical research prototype` · `Implemented` · `Superseded by Brainer` · Last substantive evidence: `2025-07-27`

**Started to ask:** Can stable concept anchors such as grammar or emotion organize word vectors through simple geometric updates?

**What the artifacts establish:** Random unit anchors can be created, word embeddings can be moved toward them, PCA can visualize the resulting geometry, and two small unit tests cover those mechanics.

**Lessons:** Moving random vectors toward random anchors does not establish learned semantics. The experiment needs seeded runs, real data, an objective, baselines, and evaluation. Empty notebooks/model files and a `requirements.txt` that is actually a long design transcript also show why reproducible project boundaries matter.

**Why it paused:** The design transcript explicitly introduces the broader Brainer vision, and later folders implement that direction. Anchor Model is best read as its geometric precursor, not as an independently abandoned product.

**Key evidence:** `anchoring_ai_project/core/anchor_space.py`, `anchoring_ai_project/core/anchored_embedding.py`, `anchoring_ai_project/experiments/`, `anchoring_ai_project/tests/`, the mislabeled `anchoring_ai_project/requirements.txt`

## Brainer

`Research lineage` · `Executed` · `Archived after rapid redesigns` · Last substantive evidence: `2025-07-29`

**Started to ask:** Can a memory-first system with permanent concepts, volatile state, editable relations, curiosity, and interaction-driven learning replace parts of a conventional language model?

**What the artifacts establish:** Three rapid versions explored fixed and volatile memory, pattern blocks, connection-vector attention, vector fusion, custom tokenization, and intent/essence response modeling. V0.1 trained on ten subject/verb/object triples; V0.12 produced a 15.96 MB checkpoint; DailyDialog was acquired for later work.

**Lessons:** Early versions learn mappings among random vectors rather than meaning. Later versions contain concrete training defects: duplicated model definitions, attention weights applied twice, gradients not cleared, final-token-only training, and a one-token CLI. V0.13 also lacks required artifacts and has fragile JSONL loading.

**Why it paused:** The architecture changed repeatedly over four days without converging on a stable measurable objective. Its ideas reappear in Dynamic-QKV, CGI, HoloBrain, and Jarvis, so the folder is more useful as lineage than as a finished claim.

**Key evidence:** `V0.1/Brain/`, `V0.12/`, `V0.13/`, DailyDialog archive

## CGI — Five-Engine Cognitive Architecture

`Research prototype` · `Implemented` · `Paused after data/trainer construction` · Last substantive evidence: `2025-12-12`

**Started to ask:** Can comprehension, reasoning, knowledge, personality, and self-review be separated into five specialized cognitive engines?

**What the artifacts establish:** The Input Engine was implemented as a GPT-style causal model that maps requests into structured goal/tone/detail/mood/topic/focus fields. The workspace contains 1,000 intent combinations, roughly 98k raw synthetic records, a separate cleaned snapshot, repair logic, isolated batching, masking, evaluation intervals, gradient accumulation, clipping, and checkpoint handling.

**Lessons:** Synthetic volume is not validation: thousands of records still miss topic/focus fields, and raw/clean record counts do not have one-to-one provenance. Only the Input Engine exists, default training paths point to absent files, and no checkpoints, metrics, or trained examples remain.

**Why it paused:** The artifacts stop after data generation and trainer repair, before the other four engines or a successful recorded run. The large architecture likely needed a smaller end-to-end vertical slice.

**Publication boundary:** Unsafe local credential storage was detected. Sensitive material is excluded; source release requires verified rotation and a history-aware scan.

**Key evidence:** `Readme.md`, `io.md`, `intent_combo.json`, `generate_openai_samples.py`, `clean_data_jsonl.py`, `input_engine/`

## chaosformer — ChaosFormer

`Architecture sketch` · `Fragment` · `Archived sibling prototype` · Last substantive evidence: `2025-08-25`

**Started to ask:** Can per-token top-k routing across multiple attention experts produce useful specialization while a balancing term prevents expert collapse?

**What the artifacts establish:** Expert attention, learned gating, a balancing-loss hook, minimal trainer/tokenizer paths, and top-k/top-p chat code were written. There is no chaos-theory mechanism; “Chaos” is a project name.

**Lessons:** The retained model is not yet a valid autoregressive experiment. It passes a batch-by-sequence padding mask as an attention mask, has no causal mask, and its optional Hugging Face Trainer sends `labels` to a model expecting `targets`. Specialization therefore remains untested.

**Why it paused:** All six source files form a same-day scaffold with no README, tests, dependencies, checkpoint, data, or results. It appears to be one member of the same package-generation burst as FractalFormer and WildFormer.

**Key evidence:** `chaosformer/config.py`, `chaosformer/core.py`, `chaosformer/trainer.py`, `chaosformer/chat.py`, `chaosformer/tokenizer.py`

## Datasets

`Shared infrastructure` · `Support` · `Frozen data archive` · Last substantive evidence: `2025-08-17`

**Started to support:** Local pretraining, instruction-tuning, and evaluation across the surrounding model experiments.

**What the artifacts establish:** Roughly 19.5 GB of OpenWebText, C4, children’s stories, Wikipedia-related files, prompt/response data, acquisition scripts, and conversion utilities were collected. Rated JSONL retains 35,331 train and 1,789 validation records.

**Lessons:** Provenance, licenses, checksums, and a dataset manifest are missing. Several Wikipedia outputs are empty or effectively empty; one patched extractor reads the whole decompressed dump into memory and does not implement its advertised multiprocessing/output behavior cleanly.

**Why it is frozen:** This is an asset pool, not a deserted application. Future value comes from a manifest and reproducible download/build scripts—not committing the corpora to GitHub.

**Key evidence:** `gemini_qa_generator.py`, `run_wikiextractor.py`, `Extractor/`, root corpus files

## Dynamic-QKV-Model — Dynamic QKV Model

`Architecture prototype` · `Implemented` · `Paused before experiment stage` · Last substantive evidence: `2025-08-16`

**Started to ask:** Can frozen core token identities generate token-specific Q/K/V gates while a separate meaning embedding remains adaptable and a write head performs targeted semantic updates?

**What the artifacts establish:** A smoke path produced correctly shaped logits and attention, preserved the frozen core, and propagated gradients into the gate network. Trainer, tokenizer, chat, save/load, and causal masking paths also exist.

**Lessons:** Tensor mechanics are not model-quality evidence. The write head is never connected to the model or trainer; the architecture is one attention layer without residuals, normalization, or FFN; argmax generation makes temperature ineffective. The monolithic and modular implementations also duplicate one another.

**Why it paused:** The code was produced in a short burst without a dataset, checkpoint, metric, test suite, README, or generated sample. It remained a reference implementation and its fixed/adaptive representation ideas continued elsewhere.

**Key evidence:** `dynamic_qkv_model.py`, `core.py`, `config.py`, `trainer.py`, `chat.py`, `tokenizer.py`

## fractalformer — FractalFormer

`Architecture sketch` · `Implemented` · `Archived sibling prototype` · Last substantive evidence: `2025-08-25`

**Started to ask:** Can several contextual “scales” run in parallel and exchange information through learned cross-scale gates?

**What the artifacts establish:** Parallel streams, per-stream attention, learned cross-stream mixing, averaged output, and minimal trainer/chat paths exist.

**Lessons:** The streams begin as identical clones and receive no downsampling or scale-specific input, so the retained code does not establish multiscale behavior. Specialization requires an actual scale mechanism plus utilization probes and ablations.

**Why it paused:** Original and updated snapshots differ mainly in tokenizer support; no README, tests, dependencies, checkpoint, or results followed. Later HoloBrain/Fusion work provided more concrete controlled experiments.

**Key evidence:** `fractalformer/core.py`, `fractalformer_updated/fractalformer/core.py`, paired trainer/chat/tokenizer files

## fusiongrammar_mod_v2 — FusionGrammar Mod V2

`Research branch` · `Implemented` · `Paused before empirical run` · Last substantive code evidence: `2025-08`

**Started to ask:** Can latent grammar codes, Gumbel-softmax routing, sparse head selection, shared bidirectional grammar layers, and a later causal phase produce explicit linguistic structure and conditional computation?

**What the artifacts establish:** A sophisticated two-phase PyTorch architecture and training path were implemented. No checkpoint or result establishes that latent codes, routing, or staged training help.

**Lessons:** Sparse routing needs utilization statistics, stability checks, matched baselines, and ablations. Architectural complexity cannot substitute for run evidence. The retained README is an unrelated personal chat transcript and is not suitable for publication.

**Why it paused:** Core code stopped in 2025; later notes and papers were added without integration. The branch appears to have yielded to the measured FusionFormer line and then the current AI Research Lab.

**Key evidence:** `fusiongrammar_mod/config.py`, `fusiongrammar_mod/model/`, `fusiongrammar_mod/train.py`, tokenizer notes

<a id="fusionformer"></a>

## Fusion_concat — FusionFormer

`Research project` · `Measured` · `Completed prototype / paused after comparison` · Last substantive evidence: `2026-04-28`

**Started to ask:** Does separating masked grammar processing from causal meaning processing, with gated bidirectional cross-links, improve language modeling over a plain GPT?

**What the artifacts establish:** This is the best-documented Fusion branch: code, checkpoints, a plain-GPT baseline, fine-tuning modes, and an intern handoff package. Authored reports record 28.22% LM accuracy and 95.20 perplexity for FusionFormer on a TinyStories GPT-4 held-out slice, versus 29.13% and 86.67 for GPT; a raw evaluation artifact is not retained.

**Lessons:** The extra architecture activated its gates but did not beat the simpler baseline in this documented comparison. The models were not parameter matched—approximately 32.42M versus 22.74M parameters—so a fairer matched comparison remains future work. Same-data evaluation, clean data, and honest negative results were still more informative than architectural complexity alone.

**Why it paused:** The research and handoff reached a natural decision point: an inconclusive/negative comparison rather than a production target. The next step would require redesigned hypotheses and controlled ablations, not more unstructured training.

**Key evidence:** `FusionFormer_V1/`, `GPT_Baseline_V1/`, checkpoints, authored evaluation reports, `INTERN_ONBOARDING.md`

## Fusion_crosslink — ConceptLM

`Research side branch` · `Implemented` · `Superseded` · Last substantive evidence: `2025-11-10`

**Started to ask:** Can fixed symbolic token identity be separated from a trainable meaning representation, then recombined with attention-pooled context and FiLM conditioning?

**What the artifacts establish:** A coherent model, data loader, trainer, tied-output path, and generator were implemented. Despite the folder name, this is not the bidirectional CrossLink design used by FusionFormer.

**Lessons:** Representation separation needs probing and controlled comparisons. The configured corpus is absent and no checkpoint, test, or metric supports the central hypothesis.

**Why it paused:** Code was written in August and documented in November, but no run followed. The branch appears to have been overtaken by the better-instrumented FusionFormer work.

**Key evidence:** `My_Idea.py`, refactored config/model/train/chat files, root `README`

## Game — Logan Engine / Emergent NPC World

`Game preproduction` · `Fragment` · `Paused before runtime integration` · Last substantive evidence: `2025-07-24`

**Started to ask:** How could a persistent RPG society model NPC needs, memory, relationships, jobs, trade, regions, magic, and possible Jarvis-driven behavior?

**What the artifacts establish:** Detailed Unreal-style C++ structures, schemas, generated NPC/world records, economy data, and an integration diagram show substantial domain modeling.

**Lessons:** Rich schemas do not prove a playable system. A small compiled vertical slice should precede expanding simulation scope. One purported JSONL file is a fenced JSON array rather than valid JSONL, illustrating the need for format validation.

**Why it paused:** There is no `.uproject`, Unreal module, build metadata, or gameplay runtime. Work stopped at preproduction/data-model design.

**Key evidence:** NPC/world C++ fragments, JSON/CSV/XLSX schemas and samples, handwritten integration diagram

## Gem_model — Early Triple-Vector Model

`Research precursor` · `Implemented` · `Superseded by later Fusion work` · Last substantive evidence: `2025-08-05`

**Started to ask:** Can separate grammar and meaning embeddings, gated cross-links, MLM/LM objectives, orthogonality, and sparse gates form a trainable dual-stream language model?

**What the artifacts establish:** The architecture and training pipeline were written and roughly 4.4 GB of corpora were collected. No successful checkpoint or evaluation remains.

**Lessons:** Corpus acquisition is not model completion. The configured `story.txt` is absent, other corpora are not cleanly wired in, and reproducibility/test metadata is missing.

**Why it paused:** The same Triple-Vector idea reappears in `GPT_model` and later becomes the measured `Fusion_concat` project. Gem is best treated as the first implementation, not a separately failed endpoint.

**Key evidence:** `Gem.py`, refactored core/trainer files, C4 and `large_corpus.txt`

## GPT_model

`Mixed research archive` · `Fragment` · `Split into later lineages` · Last substantive evidence: `2025-08-10`

**Started to ask:** The root continued the Triple-Vector grammar/meaning model on TinyStories; the separate `brain/` sequence explored memory banks, retrieval, rehearsal, and continual-learning mechanisms.

**What the artifacts establish:** TinyStories ingestion and vocabulary construction ran, and fourteen evolving brain prototypes preserve a wide range of memory designs. Empty checkpoint directories provide no evidence of a completed model run.

**Lessons:** Rapid architectural iteration is valuable for ideation but needs version boundaries, tests, saved results, and explicit evaluation criteria. Later brain versions retain TODOs and placeholders.

**Why it paused:** The root architecture progressed into FusionFormer, while the memory/rehearsal ideas appear in HoloBrain and Jarvis. The folder became an archive of forks rather than one coherent project.

**Key evidence:** root Triple-Vector files, 1.9 GB TinyStories corpus, generated vocabulary, `brain/brain_*.py`, empty checkpoint folders

## holobrain_pp — HoloBrain++ Training Lab

`Research project` · `Executed` · `Archived after large-scale experiments` · Last substantive evidence: `2025-09-21`

**Started to ask:** Can sparse experts, memory, thalamic gating, compartmental routing, and cerebellar-style correction improve a brain-inspired language model?

**What the artifacts establish:** Long training occurred: multiple roughly 3.1 GB checkpoints, including at least three distinct final-step saves plus a best-perplexity save, alongside 17,037 metric rows, plots, reports, and activation comparisons. Stored batch perplexity fell from extreme initial values to roughly 105 near the end. A MixLU rerun slightly improved validation loss over SiLU but was about 32% slower.

**Lessons:** The training/checkpoint pipeline is real, but minimum individual-batch perplexity is not a held-out benchmark. Neuroscience-inspired modules still need a plain baseline and controlled ablations; quality/throughput tradeoffs matter.

**Why it paused:** Compute/storage cost, multiple diverging versions, and limited causal attribution are visible. Later work moved toward the simpler baseline-driven FusionFormer experiments.

**Key evidence:** `HB_Train/No_stack_holo/`, multiple checkpoints, `metrics.csv`, reports, plots, activation experiments

<a id="ai-research-lab"></a>

## inverted_model — AI Research Lab / Token MOD + ATE

`Flagship research repository` · `Measured` · `Current` · Latest commit evidence: `2026-07-29`

**Started to ask:** Can a smaller frozen pretrained model gain capacity through token modulation and stage-aware expansion while retaining its prior knowledge?

**What the artifacts establish:** This is an active, version-controlled lab with tests, baselines, sealed datasets, evidence classification, checkpoint migration, and recent adaptive expansion work. Shared MOD variants reached best reported perplexity around 3.97 versus 3.52 for a Pythia-2.8B full fine-tune in one comparison. On the locked 75k UltraChat setup, a full fine-tune of pretrained Pythia-2.8B reached 3.2402, fresh ATE h1/l1 reached 3.4602, and sequential h1/l2 briefly improved to 3.4358 before overfitting. Model sizes and training steps differ, so these are same-data/evaluation comparisons rather than matched-budget results.

**Lessons:** Earlier GPT-2 results were correctly invalidated after future-token leakage was found. Lower teacher-forced perplexity did not always produce better chat. Retention tests, generation evaluation, no-expansion continuation, matched hardware, and multiple seeds remain necessary.

**Current state:** Not deserted. The clean nested Git repository tracks a remote branch and explicitly documents the pivot from expensive from-scratch Fusion models to controlled adaptation of pretrained Pythia models.

**Key evidence:** nested `AI-Research-Lab/` repository, experiment/evolution documents, tests, locked datasets, result tables, checkpoint migration utilities

## Jarvis — Jarvis / AB Artificial Brain

`Continual-learning research` · `Executed` · `Paused with unfinished run` · Last substantive evidence: `2026-05-26`

**Started to ask:** Can local plasticity, persistent memory, sparse graph processing, developmental training, and governed/rollback-safe updates support continual learning outside a conventional Transformer-only design?

**What the artifacts establish:** Extensive executable scaffolding, tests, stored states, diagnostics, synthetic benchmarks, update masks, drift budgets, rollback, and recovery checks exist. A reported governance score such as 97.07 is a rubric maturity estimate—not task accuracy.

**Lessons:** Temporal context matters for credit assignment; evaluation must not consume training randomness; storage and routing need separate tests; governance can reduce interference while also blocking useful learning. Proxy metrics must not be presented as capability.

**Why it paused:** The latest TinyStories developmental run stops at epoch 1/3, source 37/128, with next-token accuracy around 0.20 and no generated-answer support. Current reports identify restoration/rollback as the dominant failure, and the workspace contains many experimental versions plus roughly 940 MB of runtime state.

**Key evidence:** `Own Mind/`, tests, stored states, diagnostics, benchmarks, governance reports, unfinished TinyStories run

## Jarvis - RAG — Jarvis FlowLM and External-Memory Experiments

`Research/data archive` · `Executed` · `Paused after disconnected experiments` · Last substantive evidence: `2025-10-24`

**Started to ask:** Can a deep, narrow, mobile-oriented language model learn conversational “flow” while external memory supplies factual recall?

**What the artifacts establish:** A 384-dimensional, 48-layer decoder with RMSNorm, NTK-scaled RoPE, tied embeddings, checkpointing, LoRA, alternate embedding vectors, SFT pipelines, WordPiece/MiniLM studies, and a separate GPT-Neo plastic-memory/RAG package were explored. Retained generations show that SFT improved assistant-like phrasing, but output remained repetitive and incoherent.

**Lessons:** Greater depth and more data did not remove degeneration. The active FlowLM explicitly disables its memory adapters, old memory methods reference undefined state, and the working GPT-Neo RAG mechanics live in a separate package. The folder therefore demonstrates several model and retrieval components—not an integrated RAG assistant.

**Why it paused:** No model checkpoint, test suite, dependency manifest, license, or usable root README remains; most of the 6.84 GB folder is downloaded data. Training cost, incoherent generations, and later Jarvis/RAG branches are plausible reasons it stopped.

**Publication boundary:** Unsafe local credential storage was detected. Sensitive material is excluded; source release requires verified rotation and a history-aware scan.

**Key evidence:** `Jarvis/model.py`, SFT/training paths, GPT-Neo memory adapter package, `Results/Jarvis_v1.md`, `Results/Jarvis_SFT*.md`, dataset archive

## Jarvis-all — Jarvis Evolution Archive

`Historical umbrella archive` · `Executed` · `Superseded` · Last substantive evidence: `2025-07-29`

**Started to ask:** This folder preserves four successive directions: a voice/web/Arduino assistant, a dynamically expandable-vocabulary Transformer, a trained mini language model, and a later from-scratch “Brainer” GPT.

**What the artifacts establish:** Speech recognition/TTS, Flask UI, symbolic memory, intent routing, search, emotion detection, Arduino sensing, a saved light classifier, expandable embeddings/output heads, tokenizers, trainers, and small checkpoints were all implemented. `Jarvis_mini.pt` and `LLM/model.pt` prove the mini-model paths trained.

**Lessons:** Replacing embedding parameters during vocabulary expansion can preserve tensor rows while breaking optimizer continuity; the strong “no forgetting” claim lacks a retained evaluation. The latest tiny checkpoint loads but produces character-like gibberish, showing that plumbing and useful conversation quality are different milestones.

**Why it paused:** The archive bundles roughly 1.82 GB of virtual environments, fragmented paths, incompatible generations, and no top-level guide. It was superseded by more focused Jarvis folders.

**Publication boundary:** Unsafe local credential storage was detected. Sensitive material is excluded; source release requires verified rotation and a history-aware scan.

**Key evidence:** `Jarvis 0/`, `OP_Jarvis/`, `Jarvis 2.0/`, `LLM/`, `Jarvis_mini.pt`, `LLM/model.pt`

## Jarvis-flow — Flow Tokenizer Prototype

`Tokenizer experiment` · `Fragment` · `Paused as a single-file study` · Last substantive evidence: `2025-11-14`

**Started to ask:** Can capitalization be removed from base token identity and carried as a separate feature alongside span-aware BPE tokens?

**What the artifacts establish:** The file trains corpus BPE merges, preserves original-character spans, encodes/decodes casing, and builds placeholder token vectors. Its demo represents mixed-case positions correctly before encountering a Windows terminal Unicode-printing failure.

**Lessons:** Concatenating uppercase positions into a decimal float becomes ambiguous after position 9 and is vulnerable to zero/precision loss. Most vector dimensions are placeholder sequences rather than learned features.

**Why it paused:** There is no serializer, vocabulary-ID layer, test, README, dependency file, model integration, or saved result. Later capitalization work appears more rigorously in AB Former.

**Key evidence:** `tokenizer.py`

## Learnings — FusionFormer NCR-FFN Sandbox

`Architecture sandbox` · `Fragment` · `Paused with broken run paths` · Last substantive evidence: `2025-09-07`

**Started to ask:** Can a physics-inspired “Neutron Chain Reaction” FFN iteratively route between two branches and halt computation through a decaying key?

**What the artifacts establish:** The dynamic FFN, a one-block FusionFormer, step/key instrumentation, C4/local training scaffolds, checkpoint hooks, generator, and 500 conversational examples were assembled.

**Lessons:** Hard `argmax` routing is non-differentiable, while batch-averaged `.item()` stopping is neither token-adaptive nor compiler-friendly. More seriously, the attention path is non-causal during next-token training, allowing future-token leakage.

**Why it paused:** No checkpoint remains; the self-test passes an unsupported argument, one trainer references a missing config field, and data paths assume a Colab Drive layout. It stayed a learning sandbox rather than a runnable experiment.

**Key evidence:** `model.py`, training/generation scripts, `data.jsonl`, configuration files

## LexiFormer — Model Lab

`Research archive` · `Executed` · `Paused after several small-model branches` · Last substantive evidence: `2025-09-05`

**Started to ask:** Can grammar/meaning separation, compact GPTs, or learned “core” slots guide better answer generation?

**What the artifacts establish:** Four branches—Asterion, Asterion V0.1, FusionFormer, and MiniGPT—have implemented trainers/tokenizers and real checkpoints ranging from roughly 403k to 32.2M parameters. MiniGPT produced a memorized-style “my name is jarvis” response; Asterion produced word salad, and V0.1 produced fluent but irrelevant advice.

**Lessons:** The original Asterion sees target tokens while predicting the same positions, permitting trivial copying. V0.1 places its `[CLS]` core first under a causal mask, so the core cannot see the following question. In the FusionFormer branch, an identity vector is unused and final “fusion” is only averaged logits.

**Why it paused:** The folder aggregates unrelated branches without a README, tests, dependency manifest, unified entry point, or evaluation report. Checkpoints prove runs, while the retained samples honestly show that useful dialogue did not follow.

**Key evidence:** Asterion branches, `FF/`, MiniGPT files, checkpoints, DailyDialog data

## LoganBrain — Symbolic Cognitive Architecture

`Functional prototype` · `Implemented` · `Completed phase demo` · Last substantive evidence: `2025-11-27`

**Started to ask:** Can an interpretable symbolic pipeline combine parsing, concept memory, modular reasoning, planning, and follow-up resolution without learned model weights?

**What the artifacts establish:** A single standard-library file implements semantic parsing, a concept graph, query classification, fact/explanation/style modules, a MeaningTree, sentence planning, template generation, short-term/thematic memory, and LLM-client abstraction. Its four-phase demo runs and resolves “Explain more” back to the prior Einstein topic.

**Lessons:** Rule-based cognition is transparent and debuggable but narrow; outputs include tautologies such as “einstein is einstein.” The included LLM client is only an echo stub and all knowledge is process-local.

**Why it paused:** The 63 KB file is a successful phase demonstrator, not yet a maintainable application. Persistence, tests, package structure, documentation, UI, and a real model client are absent.

**Key evidence:** `logan_brain.py`

## model_artifacts — Shared Tokenizer Artifacts

`Shared infrastructure` · `Support` · `Orphaned static assets` · Last substantive evidence: `2025-08-06`

**Started to support:** Reuse of a custom 68,180-entry word vocabulary and GPT-2 BPE files across local language-model experiments.

**What the artifacts establish:** The tokenizer assets exist, including reserved tokens and separate pad/mask additions. They do not establish model performance.

**Lessons:** Tokenizer artifacts need provenance, source-data licensing, versioning, hashes, and an explicit consuming-model compatibility contract. `optimized_GPT_model` intends to load the vocabulary but its default relative path points to a nonexistent nested location.

**Why it is frozen:** The assets may be complete for their original consumer, but no README or code identifies that contract.

**Key evidence:** `vocab.txt`, `gpt2/tokenizer.json`, GPT-2 vocab/merges/config files

## my-trading-gui — Groww Multi-Symbol Trading Dashboard

`Full-stack trading prototype` · `Implemented` · `Unsafe/inoperable for live orders` · Last substantive evidence: `2025-07-14`

**Started to ask:** Can a React/Flask dashboard manage several Groww trading bots with technical signals, risk sizing, long/short state, order tracking, exits, alerts, charts, exports, statistics, and backtesting?

**What the artifacts establish:** The UI, API, threaded workers, SMA/RSI/MACD/volume analysis, position sizing, limit-order lifecycle, profit/stop/trailing exits, Telegram integration, charting, CSV export, and historical-backtest paths were designed and largely implemented. It does not establish live execution or profitability.

**Lessons:** `groww_api.py` uses an undefined `groww_valid_ity` name when constructing orders, sending every live order attempt to the exception path. The backend’s credential/config exposure and permissive cross-origin policy make deployment unsafe. Backtests omit credible fees/slippage, and global mutable thread state lacks locking.

**Why it paused:** There are no retained logs or backtest reports, no Python dependency manifest/tests, no built frontend, and the folder embeds both `node_modules` and a virtual environment. Newer `trading_bot` and `Algo Trader` work supersede it.

**Publication boundary:** Unsafe local credential storage was detected. Sensitive material is excluded; source release requires verified rotation and a history-aware scan.

**Key evidence:** React source, Flask backend, `main.py`, `groww_api.py`, chart/backtest/export paths

## optimized_GPT_model — FusionFormer Dual-Stream Prototype

`Research prototype` · `Implemented` · `Superseded before training` · Last substantive evidence: `2025-08-05`

**Started to ask:** Can parallel grammar and meaning Transformer stacks exchange information through gated bidirectional cross-links while training with masked- and causal-language objectives?

**What the artifacts establish:** Dual embeddings/stacks, MLM and CLM heads, orthogonality/gate losses, streaming data preparation, AMP/checkpoint training, tokenizer, and generation code exist. A reduced smoke forward pass produced correctly shaped logits and initialized gates around 0.119.

**Lessons:** The “MLM” stream is causal and cannot use right context as conventional MLM does. Both objectives receive the same masked input, so the causal meaning stream learns from corrupted sequences. No probe or ablation establishes stream specialization.

**Why it paused:** Default execution cannot find its intended sibling vocabulary and then exits because `story.txt` is absent. No dataset, checkpoint, result, README, dependency manifest, or test followed; later Fusion branches repaired and measured the idea.

**Key evidence:** `core_model.py`, `trainer.py`, `main.py`, `generate.py`, tokenizer/config/data files

## Project docs — Experimental Architecture Papers

`Design archive` · `Support` · `Static concept collection` · Last substantive evidence: `2025-09-05`

**Started to support:** Documentation of FusionFormer and speculative descendants: FusionFormer+MoE, HoloBrain++, FractalFormer, WildFormer, ChaosFormer, and TDS_FF.

**What the artifacts establish:** Thirteen DOCX/PDF files demonstrate architecture ideation and technical communication around dual streams, sparse experts, brain compartments, multiscale attention, controlled chaos, and dream simulation.

**Lessons:** Across the readable papers there are no references, experiments, benchmarks, accuracy/perplexity results, or links to code. Their claims must remain proposals or hypotheses. Several files are Word/PDF duplicates and `TDS_FF.pdf` lacks a useful extractable text layer.

**Why it is frozen:** Named implementation folders now preserve partial descendants. This directory is best maintained as an indexed design archive rather than presented as validated research.

**Key evidence:** FusionFormer and descendant DOCX/PDF files

## Sapient_model — Hierarchical Perception/Cognition LM

`Research prototype` · `Implemented` · `Paused before training` · Last substantive evidence: `2025-12-10`

**Started to ask:** Can nested low-level perception updates and high-level cognition updates give an autoregressive language model a useful iterative reasoning structure?

**What the artifacts establish:** A coherent PyTorch model, GPT-2 tokenizer adapter, text dataset, warmup/decay training loop, validation path, checkpoint save, and top-k/top-p generator were implemented. Low-level causal attention is repeatedly conditioned by a high-level summary state.

**Lessons:** The high-level state is initialized from a sequence mean and broadcast back into every position; without a reasoning benchmark or a plain Transformer baseline, the extra loops establish computation rather than reasoning. Iterative steps also multiply training cost.

**Why it paused:** No train/validation text, checkpoint, metric, README, dependency manifest, or test remains. The folder is a code-complete architecture sketch that never reached empirical evaluation.

**Key evidence:** `model.py`, `data.py`, `train.py`, `generator.py`, `tokenizer.py`, `config.json`

## Softmax_NO_Model — Softmax-Free Semantic Compatibility Model

`Objective/tokenizer experiment` · `Implemented` · `Paused at objective design` · Last substantive evidence: `2025-12-16`

**Started to ask:** Can next-token learning treat candidates as independent semantic compatibility scores instead of forcing them to compete through a vocabulary softmax?

**What the artifacts establish:** A documented compatibility MLP, positive-only `-logsigmoid` objective, training loop, and custom whitespace-first subword tokenizer with explicit capitalization scalars were implemented. The tokenizer includes greedy fallback, Unicode handling, round-trip APIs, and an interactive tester.

**Lessons:** With only positive targets and no negatives, the loss has a trivial direction: raise every observed compatibility logit without learning how to reject or rank alternatives. That prevents the retained objective from establishing useful generation. Several Transformer-like config fields are also reserved but unused by the mean-pooled MLP.

**Why it paused:** No training corpus, checkpoint, metric, test result, or candidate-ranking/generation evaluation remains. The project documents the idea clearly but stops before solving its identifiability problem.

**Key evidence:** `README.md`, `model.py`, `train.py`, `Tokenization/tokenizer.py`, `config.json`

## Split-Model — Two-Stream Parameter Update Model

`Training-mechanism experiment` · `Implemented` · `Paused before valid router training` · Last substantive evidence: `2025-08-14`

**Started to ask:** Can lexical memorization be isolated in token embeddings while abstract/generalizable updates are routed into Q/K/V weights, reducing interference during learning?

**What the artifacts establish:** An explicit Transformer, separate embedding/QKV parameter groups, independent optimizers, demo training/chat code, and a learned `WriteRouter` interface were implemented.

**Lessons:** The router probabilities are converted to Python numbers with `.item()` and then used only to rescale parameter gradients after backpropagation. The language-model loss therefore has no differentiable path into the router, so stepping its optimizer does not teach routing. From-scratch runs that freeze the LM head, feed-forward layers, and norms also cannot test the intended specialization cleanly.

**Why it paused:** There is no dataset, checkpoint, metric, test, README, or dependency file. The next required step was not more training but a differentiable routing objective and measured forgetting/generalization baselines.

**Key evidence:** `two_stream_transformer.py`, `wireroute_step.py`, `trainer_auto_router.py`, `train_with_auto_router.py`, `trainer_generator_chat.py`

## Story_web — StorySite

`Product prototype` · `Implemented` · `Paused at single-file frontend` · Last substantive evidence: `2025-08-17`

**Started to ask:** Can an audio-first social product make it easy to record and share five-minute life stories?

**What the artifacts establish:** One React component implements demo login, browser audio recording, a five-minute limit, profiles, an explore feed, likes, comments, sharing, hash routing, local metadata persistence, and IndexedDB audio storage.

**Lessons:** Local browser state is a strong interaction prototype but cannot support real multi-user identity, moderation, cross-device media, or social synchronization. MediaRecorder requires a secure origin, and a production path needs upload/storage policy, backend APIs, access control, tests, and object-URL lifecycle handling.

**Why it paused:** The project contains only `app.js`: no package manifest, build shell, styles configuration, backend, tests, or deployment metadata. It reached the UX-demo boundary and stopped before product infrastructure.

**Key evidence:** `app.js`

## Tokenizer

`Placeholder` · `Fragment` · `Empty`

**Started to ask:** No recoverable evidence remains.

**What the artifacts establish:** The folder is empty, so it proves only that a tokenizer work area was reserved.

**Lessons:** Empty placeholders lose intent. Even a two-line README naming the parent experiment and intended tokenizer would preserve provenance.

**Why it paused:** There are no files, timestamps from substantive artifacts, or relationships that can be established safely. Tokenizer implementations do exist inside many other project folders.

## trading_bot — Groww Trading Bot

`Trading application prototype` · `Executed` · `Paused after operational trials` · Last substantive evidence: `2025-07-27`

**Started to ask:** Can a desktop trading workstation automate multi-symbol Groww orders using moving-average, RSI, MACD, trend, and volume signals while enforcing position sizing, targets, stops, trailing exits, watchdog checks, backtests, charts, and Telegram alerts?

**What the artifacts establish:** A large Tkinter application, Groww wrapper, threaded symbol workers, risk sizing, order tracking/cancellation, backtesting, equity curves, and an alternate Flask/React-facing backend were implemented. More than 69k log lines and thousands of generated chart images prove sustained execution of polling/charting paths.

**Lessons:** Operational volume is not profitability evidence. The folder retains no concise validated performance report and its log contains repeated errors. Dependencies have drifted (`ta`, NumPy, and web-server requirements are not all captured), an entire virtual environment is embedded, and generated charts/logs grew without retention controls. Paper/live separation, broker mocks, idempotent order tests, and explicit kill switches should precede further real-order use.

**Why it paused:** The last logs end in July 2025 amid a large monolithic codebase and substantial generated output. Integration reliability and maintenance burden are better-supported explanations than any claim about strategy profitability.

**Publication boundary:** Unsafe local credential storage was detected. Sensitive material is excluded; source release requires verified rotation and a history-aware scan.

**Key evidence:** `main.py`, `main2.py`, `groww_api.py`, `requirements.txt`, `logs/bot_log.txt`, `charts/`

## Unknown — Triple-Vector Grammar/Meaning Model

`Research prototype` · `Executed` · `Paused after initial checkpoint` · Last substantive evidence: `2025-08-08`

**Started to ask:** Can two token representations specialize into grammar and meaning streams, exchange information through gated cross-links, and train jointly with masked-LM, causal-LM, orthogonality, and gate-sparsity objectives?

**What the artifacts establish:** The full dual-stream model, custom losses, trainer, generator/chat path, GPT-2 tokenizer artifacts, and a 1.94 GB `model_step_5000.pth` checkpoint exist. Training therefore reached at least the recorded save point.

**Lessons:** A step count does not establish stream disentanglement or language quality. The two eight-layer, 512-dimensional streams produce a very large checkpoint, while no retained validation metrics, ablations, samples, or baseline explain whether the cost helped. The generic folder name also destroyed useful provenance.

**Why it paused:** Work ends shortly after the 5,000-step checkpoint, without README, dependency lock, dataset, result report, or later save. Compute/storage cost and the move into named Fusion branches are plausible, but not explicitly documented.

**Key evidence:** `core_model.py`, `trainer.py`, `config.py`, `generate.py`, `chat.py`, `checkpoints/model_step_5000.pth`

## Variants — Core-Compression and Embedding-Only Experiment Suite

`Research workspace` · `Executed` · `Archived as divergent branches` · Last substantive evidence: `2025-09-16`

**Started to ask:** How do alternative compressed-core encoder/decoder designs compare, and how much language modeling can be learned when only token embeddings are trainable?

**What the artifacts establish:** Variant A pools inputs through many learned cores and generated four roughly 1.46 GB checkpoints, reaching step 16,000. Variant B implements SDPA, NTK-scaled RoPE, encoder core pooling, and a prefix-causal decoder over generated prompt/response shards. A separate embedding-only study streams C4 while freezing the Transformer and updating selected embedding rows.

**Lessons:** The workspace contains meaningful engineering but no common evaluation tying the variants together. Variant B has no retained checkpoint. The embedding-only script freezes a randomly initialized backbone, so it cannot answer the usual question of adapting a useful pretrained frozen model. Roughly 8.1 GB of duplicate checkpoints/data also show the need for artifact retention and experiment manifests.

**Why it paused:** Three incompatible experiment directions, Google Drive/local path assumptions, large generated data, and missing README/dependency/result summaries made the folder difficult to continue as one project.

**Key evidence:** `A/model.py`, `A/train*.py`, `A/variantA_step*.pt`, `B/model.py`, `B/train.py`, `pairs_creator/`, `embedding.py`

## wildformer — WildFormer

`Architecture sketch` · `Implemented` · `Archived sibling prototype` · Last substantive evidence: `2025-08-25`

**Started to ask:** Can an explicit memory bank let Transformer tokens read from and write to a separate working-memory state at every layer?

**What the artifacts establish:** Token self-attention, token-to-memory cross-attention, memory-to-token updates, a learnable initial memory bank, training/chat code, and simple/Hugging Face tokenizer paths were implemented.

**Lessons:** The memory state is reinitialized from the same learned bank on every forward pass, so the retained model does not provide persistent memory across examples or sessions. Its advertised memory-usage regularizer is a constant zero placeholder. Padding masks are also passed through an attention-mask interface that is unlikely to accept their batched shape.

**Why it paused:** The “updated” copy mainly adds local GPT-2 tokenizer fallback; the core memory mechanism is unchanged. No README, package metadata, test, checkpoint, metric, or persistence evaluation followed.

**Key evidence:** `wildformer/core.py`, `wildformer_updated/wildformer/core.py`, paired trainer/chat/tokenizer/config files
