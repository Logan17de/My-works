# Prodapt AI / LLM Interview Preparation Plan

**Interview:** 24 Sep 2026, 7:00 PM JST  
**Target:** Prodapt Tokyo — likely SDE2  
**Source of truth:** submitted resume + AI/LLM Project Interview Study Pack + Prodapt interview research

## How to use this file

- `[x]` = explained and answered successfully in practice
- `[ ]` = still needs preparation or a final mock answer
- Keep each project compact: understand the project, then answer its 4–6 most important interview questions.
- Do not overclaim results. Use: **“The evidence establishes X, but it does not yet establish Y.”**

---

# 1. Opening / Career Story

- [ ] **Tell me about yourself** — 60-second version that connects TCS, independent research, and applied AI.
- [x] **Current TCS role** — explain model evaluation/tuning work in one clean sentence.
- [x] **Why Prodapt?** — research mindset -> real production systems, larger teams, customer impact.
- [ ] **Why move from Mechanical Engineering to AI?**
- [ ] **Independent research vs production experience** — explain the distinction honestly.

---

# 2. Token MOD on Pythia

**Memory hook:** Token-conditioned trainable capacity around a frozen Pythia; shared memory beat the tested layer-unique design under the reported protocol.

- [x] **MOD vs ATE** — explain that MOD adds modular trainable capacity around the base model, while ATE expands the Transformer itself.
- [ ] **Why Pythia?** — same GPT-NeoX family, cleaner comparison, neutral pretrained baseline.
- [ ] **How Token MOD works** — Input MOD, Output MOD, Attention MOD, FFN MOD; frozen Pythia backbone.
- [ ] **Why shared vs layer-unique tables?** — explain the experiment and the result without claiming universality.
- [ ] **Key result** — know 3.9724 vs 4.0429 and what the comparison actually means.
- [ ] **Why zero-effect initialization?** — attach MOD without immediately disturbing base behavior.
- [ ] **Why separate Input and Output MOD?** — Pythia has untied input/output weights.
- [ ] **What would you run next?** — parameter-matched LoRA, 1.4B full FT, seeds, retention, scored generation.

---

# 3. ATE — Architecture / Transformer Expansion

**Memory hook:** Grow the Transformer, but separate architecture gain from extra training budget.

- [x] **What is ATE?** — Transformer architecture expansion rather than just attaching adaptation modules.
- [ ] **How h1/l1 expanded Pythia** — one head + one Transformer layer; know the rough parameter increase.
- [ ] **Key result** — fresh h1/l1 PPL 3.4602 vs Pythia-2.8B full FT 3.2402.
- [ ] **Why h1/l2 is not a clean architecture win** — extra training tokens and schedule reset are confounds.
- [ ] **Why ATE is not PEFT** — original backbone parameters were trainable in the reported run.
- [ ] **Generalization collapse** — validation worsened while training PPL kept falling.
- [ ] **Missing causal control** — continue h1/l1 with the same extra-token budget and schedule, without adding the new layer.

---

# 4. GPT-2 Causal Pipeline Failure

**Memory hook:** Great perplexity + terrible generation -> audit the pipeline, not the model.

- [ ] **What contradiction triggered the audit?**
- [ ] **What exactly was wrong with the attention mask?**
- [ ] **Why future-token leakage invalidates perplexity?**
- [ ] **What did you do after finding it?** — invalidate old results, repair pipeline, rebuild tests.
- [ ] **Main lesson** — research validity is an engineering property.
- [ ] **Failure-story answer** — prepare 60–90 second STAR version.

---

# 5. Pattern Learners on Gemma 3 270M

**Memory hook:** Named isolated skill modules with matched placement and sequential freezing.

- [x] **Why Pattern Learners?** — test whether new task-specific capacity can be isolated without rewriting earlier modules.
- [x] **Where were learners placed?** — one large learner after the final layer vs small learners after all 18 layers.
- [x] **Why parameter-match the two designs?** — placement should be the main variable, not total capacity.
- [x] **Why zero-effect initialization?** — new learner starts neutral and does not immediately disrupt base behavior.
- [x] **How sequential skill isolation works** — train addition -> freeze it -> add/train multiplication.
- [x] **What is not proven?** — no final public placement winner; continual learning is not claimed solved.
- [ ] **Why include /base runtime mode?**
- [ ] **How would you measure forgetting properly?**

---

# 6. Tiny Word-Level Pattern Learners

**Memory hook:** Random base removes pretrained arithmetic confound; compare learner placement fairly.

- [x] **Why move from Gemma to a tiny random Transformer?** — Gemma may already know arithmetic.
- [ ] **Base architecture** — 4 layers, hidden 128, 4 heads, RoPE, pre-RMSNorm, SwiGLU, untied head.
- [ ] **Three learner placements** — standalone, pre-activation, post-activation.
- [ ] **Hypothesis behind pre-activation** — influence the SwiGLU gate before activation.
- [ ] **Hypothesis behind post-activation** — modify activated FFN features before down projection.
- [x] **Tokenizer/data confound** — validation answer tokens must exist in training; commuted pairs stay in the same split.
- [x] **What is still not proven?** — no decisive multi-seed placement winner.

---

# 7. FusionFormer

**Memory hook:** Dual grammar/meaning streams; negative result taught baseline discipline.

- [ ] **Core hypothesis** — separate grammar/structure and semantic/meaning computation.
- [ ] **Architecture** — dual streams with learned gated cross-links.
- [ ] **Key result** — FusionFormer 28.22% / 95.20 PPL vs GPT 29.13% / 86.67.
- [ ] **Why the comparison was not fully clean** — parameter counts were not matched.
- [ ] **What the negative result taught you** — complexity needs baselines and ablations.
- [ ] **Would you revisit it? If yes, how?**

---

# 8. Zet Harness / Visual AI Harness

**Memory hook:** Agent workflows are compiled programs; the runtime owns durable state, not the model.

- [ ] **Why build a harness?** — model should not own durable execution semantics.
- [ ] **Core architecture** — Graph JSON -> validation/compiler -> immutable IR -> scheduler/runtime -> durable state.
- [ ] **Why immutable IR?**
- [ ] **Document identity vs semantic/execution identity**
- [ ] **Data edges vs control edges**
- [ ] **Capability model** — request is not authority.
- [ ] **Why Node + SQLite first instead of microservices?**
- [ ] **How MCP fits into the normal tool registry**
- [ ] **Why this is still active/not “finished”**

---

# 9. Veyra Editor / Runtime

**Memory hook:** AI and humans edit the same typed source; derived animation/IK state never becomes source truth.

- [ ] **Why Veyra was created** — runtime mutation is not editable source authoring.
- [ ] **Canonical .veyra document / deterministic JSON**
- [ ] **Authored state vs evaluated state**
- [ ] **Why ownership matters** — animation/IK/constraints may own the rendered value.
- [ ] **Transactional command lifecycle** — validate -> preview -> apply -> read-back -> undo.
- [ ] **Why AI does not get a privileged backdoor**
- [ ] **Browser integration failure story** — coordinate transforms / event propagation / real-handler testing.
- [ ] **Hardest engineering problem** — correct source-of-truth ownership.

---

# 10. Agentic AI Coding Assistant

**Memory hook:** Generate-run-refine worked; secure execution became the real problem.

- [ ] **Core generate-run-refine loop**
- [ ] **Why provider abstraction?** — OpenAI / Gemini / offline stub.
- [ ] **Why use an offline stub?**
- [ ] **Why FastAPI / browser UI / desktop IDE / PyInstaller?**
- [ ] **What made local code execution unsafe?**
- [ ] **Why the project paused at demo boundary**
- [ ] **How this project led to Zet Harness**

---

# 11. FAISS / Retrieval-Memory Prototype

**Memory hook:** Retrieval code existing is not proof retrieval improves behavior.

- [x] **RAG meaning and basic flow**
- [x] **Embeddings / vectors / FAISS concept**
- [x] **Precision@K and retrieval-vs-answer evaluation basics**
- [x] **Reranking / hybrid retrieval concept**
- [ ] **What this specific prototype actually implemented**
- [ ] **Audit findings** — disconnected memory/meta path, missing valid causal objective.
- [ ] **Why you do NOT call it validated production RAG**
- [ ] **How you would rebuild it correctly**
- [ ] **Main lesson** — component presence != causal mechanism validation.

---

# 12. Dynamic-QKV

**Memory hook:** Gradient flow is not mechanism validation.

- [ ] **Core idea** — separate stable token identity from adaptable meaning.
- [x] **What dynamic Q/K/V gating was meant to do**
- [ ] **What was actually implemented**
- [ ] **What was missing** — write head not connected to training; shallow architecture.
- [ ] **Main lesson** — differentiability does not prove usefulness.

---

# 13. FractalFormer

**Memory hook:** Parallel streams are not multiscale unless computation really differs.

- [ ] **Core hypothesis**
- [ ] **Parallel-stream / cross-stream mixing design**
- [ ] **Why it was not truly multiscale**
- [ ] **What a proper follow-up would require**
- [ ] **Why no performance claim is safe**

---

# 14. ChaosFormer

**Memory hook:** Sparse routing needs a causally valid language model before it means anything.

- [ ] **Core idea** — per-token top-k expert attention.
- [ ] **Why balancing loss was included**
- [ ] **Why the retained experiment was invalid**
- [ ] **Trainer/model interface problem**
- [ ] **What is safe to claim vs avoid**

---

# 15. AB Former

**Memory hook:** Tokenizer deduplication was real; the smoke corpus could not test the efficiency hypothesis.

- [ ] **Core hypothesis** — factor casing away from lexical identity.
- [ ] **Tokenizer normalization result** — 52,696 duplicate regular tokens merged.
- [ ] **Smoke-test results** — baseline vs structured accuracy.
- [ ] **Why the benchmark could not test the central hypothesis**
- [ ] **What a proper experiment would require**

---

# 16. Transformer Fundamentals From the Resume

- [ ] **Self-attention and Q/K/V**
- [ ] **Why divide by sqrt(d_k)**
- [ ] **Causal masking**
- [ ] **Multi-head attention**
- [ ] **RoPE**
- [ ] **RMSNorm**
- [ ] **SwiGLU**
- [ ] **Pre-norm vs post-norm**
- [ ] **Untied embedding / LM head**
- [ ] **Perplexity and its limitations**

---

# 17. Fine-Tuning / Continual Learning / Evaluation

- [x] **RAG vs fine-tuning vs prompting**
- [ ] **Pretraining vs SFT vs fine-tuning**
- [ ] **Full fine-tuning vs LoRA / PEFT**
- [ ] **Catastrophic forgetting**
- [ ] **Parameter efficiency vs compute efficiency**
- [ ] **Why sealed test sets matter**
- [ ] **Deterministic splits / leakage prevention**
- [ ] **Exact-answer evaluation**
- [ ] **Ablations and matched baselines**
- [ ] **Checkpoint selection / overfitting**

---

# 18. FastAPI / REST / WebSockets

- [x] **What an API is**
- [x] **REST as API design style**
- [x] **GET vs POST**
- [x] **Why FastAPI**
- [x] **Pydantic request validation**
- [x] **async/await for I/O-bound LLM calls**
- [x] **WebSocket vs REST**
- [ ] **PUT / PATCH / DELETE / idempotency**
- [ ] **HTTP status codes that matter**
- [ ] **Timeout / retry / cancellation**
- [ ] **Streaming LLM responses**
- [ ] **Authentication vs authorization**
- [ ] **Design one LLM REST endpoint end-to-end**

---

# 19. Python / PyTorch / Hugging Face

- [ ] **List / tuple / set / dict**
- [ ] **Mutable vs immutable**
- [ ] **Decorators**
- [ ] **Generators**
- [ ] **Exception handling**
- [ ] **Context managers**
- [ ] **Threads vs processes vs async**
- [ ] **PyTorch autograd / requires_grad / freezing**
- [ ] **Optimizer parameter groups / learning rates**
- [ ] **Hugging Face model/tokenizer/checkpoint workflow**

---

# 20. SQL / Database — Prodapt Role Gap

- [ ] **SELECT / WHERE / GROUP BY / HAVING**
- [ ] **INNER JOIN vs LEFT JOIN**
- [ ] **CTEs / subqueries**
- [ ] **Indexes**
- [ ] **Primary key / foreign key**
- [ ] **Transactions / ACID**
- [ ] **ROW_NUMBER / RANK / DENSE_RANK**
- [ ] **Find duplicates / top-N per group**
- [ ] **SQL vs vector database**
- [ ] **How to debug a slow query**

---

# 21. Microservices / Production AI — Prodapt Role Gap

- [ ] **What is a microservice?**
- [ ] **Microservices vs modular monolith**
- [ ] **REST vs async messaging / queues**
- [ ] **Retries / idempotency / circuit breaker**
- [ ] **Service failure after partial write**
- [ ] **Observability: logs, metrics, traces**
- [ ] **LLM outage / fallback strategy**
- [ ] **Latency debugging**
- [ ] **Model rollout / canary / rollback**
- [ ] **Prompt injection / least privilege / tool permissions**

---

# 22. System Design / Customer Scenarios

- [ ] **Design an enterprise RAG chatbot**
- [ ] **Customer says “build an AI assistant” — clarify requirements first**
- [ ] **Customer asks for 100% accuracy**
- [ ] **RAG answers are wrong — isolate retrieval vs generation failure**
- [ ] **Service works for 5 users but slows at 500**
- [ ] **Choose large vs small model**
- [ ] **Protect customer data with external LLM APIs**
- [ ] **Autonomous agent connected to production DB**
- [ ] **Requirement changes midway**
- [ ] **Client disagrees with your recommendation**

---

# 23. Behavioral / Failure Stories

- [ ] **Future-token leakage** — primary technical failure story.
- [ ] **FusionFormer negative result** — primary negative-result story.
- [ ] **FAISS/RAG prototype audit** — prototype vs validated mechanism.
- [ ] **Veyra browser integration bugs** — integration-testing lesson.
- [ ] **Why so many projects?** — explain research progression, not random experimentation.
- [ ] **Research philosophy now** — simplest falsifiable experiment, controlled evidence, explicit claim boundaries.

---

# 24. Japanese Fluency Round

- [ ] **自己紹介**
- [ ] **現在の仕事**
- [ ] **AIの経験**
- [ ] **転職理由**
- [ ] **なぜProdaptか**
- [ ] **日本でのキャリア**
- [ ] **LLMプロジェクトを簡単に説明**
- [ ] **日本語で仕事できますか**
- [ ] Recovery phrases for repetition / slower speech / clarification

---

# 25. Final Mocks

- [ ] **Technical Round 1 mock** — CV + LLM fundamentals + Python/API/SQL
- [ ] **Technical Round 2 mock** — architecture + production AI + customer scenarios
- [ ] **Japanese fluency mock**
- [ ] **Final resume line-by-line defense**
- [ ] **Last-night 30-second project drill**

---

# Core Numbers to Know

Do not memorize a number without knowing the dataset/protocol behind it.

- [ ] Token MOD shared best: **3.9724 PPL**
- [ ] Token MOD layer-unique v2: **4.0429 PPL**
- [ ] Locked UltraChat Pythia-2.8B full FT: **3.2402 PPL**
- [ ] Fresh ATE h1/l1: **3.4602 PPL**
- [ ] Sequential h1/l2: **3.4358 PPL** — extra-training confound
- [ ] UltraChat: **72k / 1.5k / 1.5k** train/validation/sealed test
- [x] Gemma learner capacity: about **819K vs 829K**
- [ ] Tiny learner capacity: about **77K** across matched placements
- [ ] FusionFormer: **28.22% / 95.20 PPL**
- [ ] GPT baseline: **29.13% / 86.67 PPL**
- [ ] AB Former tokenizer duplicates merged: **52,696**

---

# Final Rule

When uncertain, answer:

> **“The project establishes X, but it does not yet establish Y. The next control I would run is Z.”**

This keeps the interview technically honest and demonstrates research maturity.


---

# 26. AI Architecture Mastery Expansion — Beyond Resume

**Purpose:** This section is deliberately broader than the resume. It is interview-breadth preparation so that architecture questions about modern LLMs do not surprise you.

**Mastery target:** For every checked item, be able to give:
1. a 20–30 second definition,
2. why it exists,
3. one trade-off,
4. one concrete example or use case.

## 26.1 Transformer End-to-End

- [ ] **Walk through a decoder-only Transformer from token IDs to next-token logits.**
- [ ] **Encoder-only vs decoder-only vs encoder-decoder — when is each used?**
- [ ] **What happens inside one Transformer block?**
- [ ] **Why do residual connections matter?**
- [ ] **Why do we normalize activations?**
- [ ] **Pre-norm vs post-norm — what changes for training stability?**
- [ ] **What is the FFN/MLP doing that attention does not do?**
- [ ] **Why are modern FFNs often much wider than the hidden size?**
- [ ] **What is SwiGLU and why is it popular?**
- [ ] **What is weight tying? Why might the LM head be tied or untied?**
- [ ] **What exactly happens during autoregressive generation?**
- [ ] **Why can training be parallel across sequence positions while decoding is sequential?**

## 26.2 Attention Fundamentals

- [ ] **Explain Q, K, and V mathematically and intuitively.**
- [ ] **Derive scaled dot-product attention at a high level.**
- [ ] **Why divide by sqrt(d_k)?**
- [ ] **What does softmax do inside attention?**
- [ ] **What is a causal mask?**
- [ ] **Padding mask vs causal mask — what is the difference?**
- [ ] **Why does future-token leakage make language-model results invalid?**
- [ ] **What is multi-head attention and why multiple heads?**
- [ ] **What happens if all attention heads learn the same thing?**
- [ ] **Attention complexity with sequence length — where does O(n²) come from?**
- [ ] **What is the KV cache and why does decoding need it?**

## 26.3 Modern Attention Variants

- [ ] **MHA vs MQA vs GQA — what changes and why?**
- [ ] **Why does GQA reduce KV-cache memory compared with full MHA?**
- [ ] **What is Multi-Head Latent Attention (MLA)?**
- [ ] **How does MLA compress the KV cache conceptually?**
- [ ] **GQA vs MLA — what trade-off is each making?**
- [ ] **What is sliding-window attention?**
- [ ] **Local vs global attention — when would you mix them?**
- [ ] **What is sparse attention?**
- [ ] **What is block-sparse attention?**
- [ ] **What are attention sinks and why can they help long-context models?**
- [ ] **What is FlashAttention? Why is it faster if the mathematical attention result is the same?**
- [ ] **FlashAttention vs a new attention architecture — why are they not the same thing?**
- [ ] **What is QK normalization / QKNorm and why might it stabilize training?**

## 26.4 Positional Information and Long Context

- [ ] **Why does a Transformer need positional information?**
- [ ] **Absolute positional embeddings vs relative position methods.**
- [ ] **Explain RoPE intuitively.**
- [ ] **Why does RoPE act on Q/K rather than directly changing V?**
- [ ] **What is ALiBi at a high level?**
- [ ] **What is position interpolation / RoPE scaling?**
- [ ] **What is YaRN-style context extension conceptually?**
- [ ] **Why does extending context length sometimes hurt short-context quality?**
- [ ] **What is the “lost in the middle” problem?**
- [ ] **Long-context prompting vs RAG — when would you prefer each?**
- [ ] **How would you evaluate whether a model really uses a 100K+ context effectively?**

## 26.5 Mixture of Experts (MoE)

- [ ] **What is MoE?**
- [ ] **Dense model vs MoE model — total parameters vs active parameters.**
- [ ] **What does the router do?**
- [ ] **Top-1 vs top-2 routing.**
- [ ] **Why can MoE increase model capacity without activating every parameter per token?**
- [ ] **What is expert specialization?**
- [ ] **What is expert collapse?**
- [ ] **Why is load balancing needed?**
- [ ] **What is an auxiliary routing/load-balancing loss?**
- [ ] **What is capacity factor / expert capacity?**
- [ ] **What happens when too many tokens route to one expert?**
- [ ] **Shared experts vs routed experts.**
- [ ] **Fine-grained experts vs a few large experts.**
- [ ] **Expert parallelism — why does MoE create communication overhead?**
- [ ] **Why can an MoE model be memory-heavy even when active compute is low?**
- [ ] **Why does active-parameter count not tell the whole inference-speed story?**
- [ ] **When would you choose dense instead of MoE?**
- [ ] **Can MoE be used only in FFN layers, or elsewhere too?**
- [ ] **How would you test whether experts are actually specializing?**

## 26.6 State-Space Models, Linear Attention, and Hybrid Architectures

- [ ] **What problem are State Space Models (SSMs) trying to solve?**
- [ ] **Transformer attention vs SSM recurrence at a high level.**
- [ ] **What is Mamba conceptually?**
- [ ] **What changed with Mamba-2 / state-space duality at a high level?**
- [ ] **What is Mamba-3 trying to improve?**
- [ ] **Why can fixed-state recurrent/linear models struggle with exact retrieval?**
- [ ] **What is linear attention?**
- [ ] **Why is “linear complexity” not automatically faster on real GPUs?**
- [ ] **What is Gated DeltaNet / gated linear recurrence at a high level?**
- [ ] **Why are hybrid Attention + SSM architectures becoming attractive?**
- [ ] **Inter-layer hybrid vs intra-layer hybrid — what is the difference?**
- [ ] **When would you retain occasional full-attention layers inside an SSM-heavy model?**
- [ ] **How can MoE and hybrid Attention/SSM designs coexist?**

## 26.7 Tokenization and Embeddings

- [ ] **What is tokenization and why is it trained before model pretraining?**
- [ ] **BPE vs WordPiece vs Unigram tokenization.**
- [ ] **Byte-level vs subword tokenization.**
- [ ] **Vocabulary size trade-offs.**
- [ ] **Why can tokenizer choice affect multilingual performance and compute?**
- [ ] **What are special tokens?**
- [ ] **What is an embedding layer?**
- [ ] **Token embeddings vs sentence/document embeddings.**
- [ ] **What is an embedding dimension?**
- [ ] **Cosine similarity vs dot product vs L2 distance.**
- [ ] **Why does normalization make cosine and dot-product ranking closely related?**
- [ ] **Bi-encoder vs cross-encoder.**
- [ ] **Why is a cross-encoder usually better for reranking but slower?**
- [ ] **What is late interaction / ColBERT-style retrieval?**

## 26.8 Training Objectives and Adaptation

- [ ] **Next-token prediction / causal language modeling objective.**
- [ ] **Masked language modeling vs causal language modeling.**
- [ ] **Pretraining vs continued pretraining vs SFT.**
- [ ] **Instruction tuning vs domain adaptation.**
- [ ] **Full fine-tuning vs LoRA.**
- [ ] **What exactly does LoRA modify?**
- [ ] **Why does low rank reduce trainable parameters?**
- [ ] **LoRA vs QLoRA.**
- [ ] **Adapters / prefix tuning / prompt tuning — conceptually how do they differ?**
- [ ] **What is catastrophic forgetting?**
- [ ] **Replay / mixed old+new data as a forgetting mitigation strategy.**
- [ ] **Freezing, adapters, and selective plasticity as mitigation strategies.**
- [ ] **What is knowledge distillation?**
- [ ] **Teacher-student distillation vs self-distillation.**
- [ ] **What is DPO?**
- [ ] **RLHF at a high level — reward model + policy optimization.**
- [ ] **What is GRPO at a high level and why is it associated with modern reasoning training?**
- [ ] **Why does post-training quality depend heavily on data quality and evaluation?**

## 26.9 Reasoning and Test-Time Compute

- [ ] **What is chain-of-thought prompting?**
- [ ] **Why should internal reasoning and final answer quality be treated separately?**
- [ ] **What is self-consistency?**
- [ ] **What is best-of-N sampling?**
- [ ] **What is a verifier / reward model?**
- [ ] **What is test-time compute scaling?**
- [ ] **Why can spending more inference compute improve answer quality?**
- [ ] **When does test-time scaling become too expensive?**
- [ ] **Tool use vs reasoning in the model weights.**
- [ ] **Workflow vs agent.**
- [ ] **Function/tool calling — what actually happens outside the model?**
- [ ] **Why should authorization never be delegated to the LLM itself?**
- [ ] **Structured output / JSON schema — why useful in production AI?**

## 26.10 Advanced RAG — Retrieval

- [x] **Basic RAG pipeline**
- [x] **Dense embeddings + FAISS**
- [x] **Precision@K / Recall@K basics**
- [x] **Hybrid retrieval concept**
- [x] **Reranking concept**
- [ ] **Sparse retrieval / BM25 — how does it differ from dense retrieval?**
- [ ] **Dense vs sparse vs hybrid — when does each win?**
- [ ] **Reciprocal Rank Fusion (RRF) — why combine rankings?**
- [ ] **Query rewriting**
- [ ] **Multi-query retrieval**
- [ ] **HyDE — generate a hypothetical document before retrieval.**
- [ ] **Semantic chunking vs fixed-size chunking.**
- [ ] **Parent-child / hierarchical retrieval.**
- [ ] **Metadata filtering and ACL-aware retrieval.**
- [ ] **Late-interaction retrieval / ColBERT.**
- [ ] **Cross-encoder reranking.**
- [ ] **Contextual retrieval / adding surrounding context before indexing.**
- [ ] **GraphRAG — when relationships matter more than nearest-neighbor similarity.**
- [ ] **Multi-hop retrieval.**
- [ ] **Agentic RAG — when the system chooses retrieval steps dynamically.**
- [ ] **Multimodal RAG — text + image/audio/video/document structure.**
- [ ] **Freshness / incremental indexing / deleting stale knowledge.**
- [ ] **Deduplication and conflicting-source handling.**

## 26.11 Advanced RAG — Generation and Evaluation

- [ ] **Retrieval failure vs generation failure — how do you isolate them?**
- [ ] **Faithfulness vs answer correctness vs relevance.**
- [ ] **Precision@K vs Recall@K vs MRR vs nDCG at a high level.**
- [ ] **How do you build a labeled retrieval evaluation set?**
- [ ] **How do you evaluate groundedness?**
- [ ] **Human evaluation vs LLM-as-judge.**
- [ ] **What are the risks of using an LLM as the evaluator?**
- [ ] **When should a RAG system refuse to answer?**
- [ ] **How do citations help and what do citations NOT guarantee?**
- [ ] **How can irrelevant retrieved chunks hurt generation?**
- [ ] **How can too many chunks hurt quality?**
- [ ] **How would you choose top-K?**
- [ ] **How would you choose chunk size and overlap experimentally?**
- [ ] **How do you evaluate RAG latency and cost separately from quality?**
- [ ] **RAG cache / semantic cache — when does it help?**

## 26.12 RAG Security and Failure Modes

- [ ] **Prompt injection through retrieved documents.**
- [ ] **Data poisoning / malicious indexed content.**
- [ ] **Tenant/ACL leakage in multi-user RAG.**
- [ ] **Why filtering after retrieval can be unsafe or inefficient.**
- [ ] **PII / sensitive-data redaction.**
- [ ] **Source trust / provenance scoring.**
- [ ] **Conflicting documents — how should the system respond?**
- [ ] **Stale documents / versioning.**
- [ ] **Why the model should not treat retrieved text as trusted instructions.**
- [ ] **How to make a RAG system auditable.**

## 26.13 Modern LLM Inference and Serving

- [ ] **Prefill vs decode — what is different computationally?**
- [ ] **Why decode is often memory-bandwidth-bound.**
- [ ] **What is TTFT (time to first token)?**
- [ ] **What is inter-token latency?**
- [ ] **Latency vs throughput — why optimizing one may hurt the other.**
- [ ] **What is continuous batching?**
- [ ] **Why static batches waste GPU utilization during decoding.**
- [ ] **What is PagedAttention / paged KV-cache management?**
- [ ] **Why is prefix caching useful for chat/agent workloads?**
- [ ] **What is chunked prefill?**
- [ ] **Why can a huge prefill stall other users?**
- [ ] **What is speculative decoding?**
- [ ] **Draft model + target model verification — how does it speed decoding?**
- [ ] **When does speculative decoding help and when does it not?**
- [ ] **What is prefill/decode disaggregation?**
- [ ] **Why might production systems scale prefill and decode separately?**
- [ ] **Streaming output vs waiting for complete generation.**
- [ ] **How do you estimate concurrent users from model memory + KV cache?**

## 26.14 Quantization and Model Compression

- [ ] **FP32 vs FP16 vs BF16 vs FP8.**
- [ ] **INT8 vs INT4 quantization.**
- [ ] **Weight quantization vs activation quantization vs KV-cache quantization.**
- [ ] **Post-training quantization vs quantization-aware training.**
- [ ] **What are GPTQ and AWQ conceptually?**
- [ ] **What is GGUF / why common in local inference?**
- [ ] **Why does quantization reduce memory bandwidth pressure?**
- [ ] **Why can lower precision hurt quality?**
- [ ] **Why hardware support matters for whether a datatype is actually faster.**
- [ ] **Pruning vs quantization vs distillation.**
- [ ] **What is sparsity and why is theoretical sparsity not always real speedup?**

## 26.15 Distributed Training and Scaling

- [ ] **Data parallelism**
- [ ] **Tensor parallelism**
- [ ] **Pipeline parallelism**
- [ ] **Expert parallelism**
- [ ] **Sequence/context parallelism**
- [ ] **Why distributed training needs communication between GPUs**
- [ ] **What is FSDP / ZeRO at a high level?**
- [ ] **Why optimizer state can consume more memory than model weights during training**
- [ ] **Gradient accumulation — why use it?**
- [ ] **Gradient checkpointing — compute vs memory trade-off**
- [ ] **Mixed-precision training**
- [ ] **Gradient clipping**
- [ ] **Learning-rate warmup**
- [ ] **What causes training loss spikes / instability?**
- [ ] **Checkpointing and resume correctness in distributed training**

## 26.16 Modern Architectural Features to Recognize

- [ ] **Multi-Token Prediction (MTP) — predicting multiple future tokens as an auxiliary objective.**
- [ ] **Speculative heads / Medusa/EAGLE-style decoding at a high level.**
- [ ] **Mixture of Depths / adaptive computation — not every token necessarily uses every block.**
- [ ] **Token pruning / dynamic token selection.**
- [ ] **Shared KV / compressed KV architectures.**
- [ ] **Memory tokens / recurrent memory concepts.**
- [ ] **Retrieval-augmented language models vs very-long-context models.**
- [ ] **Diffusion language models vs autoregressive language models at a high level.**
- [ ] **Multimodal Transformers — where image/audio tokens enter the architecture.**
- [ ] **Vision encoder + projector + LLM vs native multimodal architectures.**
- [ ] **Why modern models increasingly combine several ideas rather than using one pure architecture.**

## 26.17 Architecture Comparison Drills

- [ ] **Dense Transformer vs MoE Transformer**
- [ ] **MHA vs GQA vs MLA**
- [ ] **Full attention vs sliding-window attention**
- [ ] **Transformer vs Mamba/SSM**
- [ ] **Pure Transformer vs hybrid Attention+SSM**
- [ ] **RAG vs long context**
- [ ] **RAG vs fine-tuning**
- [ ] **Full fine-tuning vs LoRA/QLoRA**
- [ ] **Vector search vs BM25**
- [ ] **Bi-encoder retrieval vs cross-encoder reranking**
- [ ] **REST vs WebSocket vs streaming HTTP**
- [ ] **Local inference vs API-hosted inference**
- [ ] **Dense deployment vs quantized deployment**
- [ ] **Single-GPU vs tensor-parallel serving**

## 26.18 Whiteboard Drills

- [ ] **Draw a decoder-only Transformer block.**
- [ ] **Draw attention with Q/K/V and causal mask.**
- [ ] **Draw GQA and explain which heads share K/V.**
- [ ] **Draw a simple MoE FFN with router -> top-k experts -> combine.**
- [ ] **Draw a classic RAG pipeline.**
- [ ] **Draw hybrid retrieval + reranker + generation.**
- [ ] **Draw an LLM serving stack with queue -> scheduler -> GPU -> KV cache -> streaming response.**
- [ ] **Draw an agent/tool loop with application-side authorization.**
- [ ] **Draw Attention + SSM hybrid model conceptually.**
- [ ] **Draw your Token MOD vs ATE distinction.**

---

# 27. Current 2025–2026 Architecture Topics — Quick Recognition List

These are **breadth topics**, not resume claims. The goal is recognition + a 30-second explanation, not pretending hands-on experience.

- [ ] **MLA (Multi-Head Latent Attention)** — compressed latent KV representation for cache efficiency.
- [ ] **MHA -> MQA -> GQA -> MLA evolution**
- [ ] **Hybrid Transformer + Mamba / SSM architectures**
- [ ] **Mamba-2 / Mamba-3 high-level evolution**
- [ ] **Gated DeltaNet / modern gated linear recurrence**
- [ ] **Shared+routed MoE experts and fine-grained expert routing**
- [ ] **Multi-Token Prediction**
- [ ] **Test-time compute / reasoning scaling**
- [ ] **Agentic tool-use systems**
- [ ] **GraphRAG / agentic RAG / contextual retrieval**
- [ ] **Late-interaction retrieval (ColBERT-style)**
- [ ] **PagedAttention / continuous batching / prefix caching**
- [ ] **Speculative decoding**
- [ ] **Chunked prefill and prefill/decode disaggregation**
- [ ] **FP8 / INT4 serving and KV-cache quantization**
- [ ] **FlashAttention-family kernels**
- [ ] **Diffusion language models**
- [ ] **Multimodal token architectures**
- [ ] **Adaptive-depth / sparse-compute architectures**

## Current-topic reading anchors

- Hybrid Attention/SSM evaluation and modern hybrid design: https://arxiv.org/pdf/2510.04800
- Mamba-3 / inference-first SSM evolution: https://arxiv.org/pdf/2603.15569
- MLA migration / KV-cache compression concepts: https://proceedings.neurips.cc/paper_files/paper/2025/file/75d13a472f570755af2a4ae4ac3d6724-Paper-Conference.pdf

---

# 28. AI Architecture “Surprise Question” Rule

If an interviewer names an architecture you have heard of but have not used:

> **“I haven’t implemented that architecture directly, but my understanding is that it changes X to solve Y. The trade-off I would want to measure is Z.”**

Do **not** fake hands-on experience. Architecture breadth is useful; evidence-backed depth on your own projects is still the priority.
