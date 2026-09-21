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
