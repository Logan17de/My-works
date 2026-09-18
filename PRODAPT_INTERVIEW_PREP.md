# Prodapt Tokyo AI SDE Interview Preparation Plan

**Interview:** 24 Sep 2026, 7:00 PM JST  
**Likely level:** AI SDE1 / AI SDE2 (profile is closer to SDE2 on experience)  
**Format confirmed by recruiter:** 3 rounds — 2 technical discussions in English + final fluency check  
**Preparation rule:** Optimize for discussion, project defense, and applied AI reasoning. Do not spend large blocks on generic Python trivia or deep event-loop internals unless directly relevant.

## Evidence-based focus

### Prodapt role requirements
AI SDE1/SDE2 emphasize:
- Python
- LLMs / applied AI
- REST APIs / FastAPI-style backend work
- SQL
- Prompt engineering
- Full-stack / integration fundamentals
- RAG and production AI patterns for SDE2

### Candidate-reported Prodapt interview themes
Recent Prodapt interview reports include:
- Python concepts, decorators, OOP
- Basic Python string/list logic
- SQL joins, duplicates, window functions
- Project architecture and end-to-end explanation
- Scenario-based problem solving
- Questions about hands-on depth rather than only definitions

This interview's recruiter specifically said **discussion only**, so prioritize explaining and reasoning aloud. Keep enough Python/SQL practice to solve simple verbal or screen-shared questions if they appear.

## Priority split

- **35% — Resume/project defense**
- **30% — LLM / RAG / applied AI**
- **15% — Python + REST/FastAPI**
- **10% — SQL**
- **10% — Japanese fluency / communication**

## Core interview stories to prepare

### 1. Applied GenAI / RAG assistant
Be ready to explain:
- Problem being solved
- Why FastAPI
- Request/response flow
- LLM provider integration
- FAISS / retrieval flow
- Embeddings and similarity search
- Prompt construction
- Memory / retrieval behavior
- Failure handling, retries, timeouts
- How to evaluate retrieval and answer quality
- What would change for production scale

Likely follow-ups:
- Why FAISS instead of a managed vector DB?
- How would you choose chunk size?
- What if retrieval returns irrelevant chunks?
- RAG vs fine-tuning?
- How would you reduce hallucinations?
- How would you secure the API?
- How would you stream LLM responses?

### 2. Pythia / Token MOD + ATE research
Be ready to explain:
- Hypothesis
- Experimental controls
- Why the benchmark was locked
- Fine-tuning vs adaptation
- Causal masking
- Why future-token leakage invalidated earlier results
- Perplexity vs free-generation quality
- Overfitting/generalization collapse
- Why negative or inconclusive results were kept

Likely follow-ups:
- Explain attention
- Why scale QK by sqrt(d_k)?
- What is causal masking?
- RoPE / RMSNorm / SwiGLU
- SFT vs pretraining vs PEFT
- How do you evaluate an LLM properly?

### 3. From-scratch Transformer / pattern learners
Use this to prove Transformer understanding beyond API usage:
- Architecture
- Token flow
- Attention
- Residual path
- Normalization
- Output head
- Leakage-aware data generation
- Parameter matching
- Why isolated learners were frozen/unfrozen

## Day-by-day plan

### Sep 18 — Project defense + targeted Python/API
**Goal:** explain your own work clearly before studying generic questions.

1. Prepare a 60–90 second end-to-end explanation of the FastAPI + RAG assistant.
2. Prepare a 60–90 second explanation of the Pythia/ATE work.
3. Python/API topics only at interview depth:
   - mutable vs immutable
   - list / tuple / set / dict
   - decorators
   - OOP basics
   - exceptions
   - generators
   - async/await for I/O-bound work
   - GET/POST/PUT/PATCH/DELETE
   - status codes
   - idempotency
   - authentication / validation
   - FastAPI + Pydantic
4. Practice 2 simple verbal Python logic questions.

**Already covered:** async/await basic mental model.  
Remember: async does not make one external call faster; it allows other work while waiting.

### Sep 19 — SQL practical
- SELECT / WHERE / GROUP BY / HAVING
- JOINs
- CTEs
- subqueries
- duplicates
- ROW_NUMBER / RANK / DENSE_RANK
- top-N per group
- indexes
- primary vs foreign key
- transactions / ACID
- SQL vs NoSQL

Practice:
- Find duplicates
- Top customer per month
- Second-highest value
- Missing IDs / rows
- Explain query aloud before writing it

### Sep 20 — LLM / Transformer fundamentals
- Attention equation and intuition
- Q/K/V
- causal mask
- context window
- tokenization
- embeddings
- RoPE
- RMSNorm
- SwiGLU
- pretraining vs SFT vs PEFT/LoRA
- temperature / top-p
- hallucination causes
- evaluation

### Sep 21 — RAG / enterprise GenAI
- ingestion
- chunking
- embeddings
- vector search
- hybrid search
- reranking
- metadata filters
- prompt grounding
- citations
- evaluation
- guardrails
- prompt injection
- access control
- caching / latency / cost

Primary system-design prompt:
**Design an enterprise RAG chatbot for internal documents.**

### Sep 22 — Production AI / system design
- FastAPI service design
- microservices basics
- synchronous vs asynchronous boundaries
- request queues / workers
- streaming responses
- retries / exponential backoff
- rate limiting
- observability
- Docker / Kubernetes concepts
- CI/CD concepts
- SQL + vector DB integration
- scaling an LLM-backed API

### Sep 23 — Full mock interview
**Round 1:** Resume + project deep dive  
**Round 2:** Applied AI / RAG / system design  
**Round 3:** Japanese fluency simulation

No studying large new topics after the mock. Fix only observed weak areas.

### Sep 24 — Interview day
- 30–45 min light review
- Review project stories
- Review attention + RAG architecture
- Review SQL patterns
- Japanese self-introduction
- No heavy new study

## High-probability technical questions

### Resume / projects
1. Tell me about one AI project you built end-to-end.
2. What exactly did you implement yourself?
3. What was the hardest technical problem?
4. What failed, and how did you debug it?
5. How did you evaluate the system?
6. What would you change for production?

### LLM / RAG
1. RAG vs fine-tuning — when would you use each?
2. What are embeddings?
3. How does vector similarity search work?
4. How would you reduce hallucination?
5. How do you evaluate a RAG system?
6. What are hybrid search and reranking?
7. How do you protect against prompt injection?
8. How would you handle stale or conflicting documents?

### Python / FastAPI
1. Why FastAPI?
2. How does request validation work?
3. What is a decorator?
4. Explain OOP concepts with a practical example.
5. What is async/await and when is it useful?
6. How would you call an external LLM reliably?
7. How would you handle timeout/retry/error responses?
8. How would you stream tokens to a frontend?

### REST
1. GET vs POST vs PUT vs PATCH vs DELETE
2. What does idempotent mean?
3. 200 vs 201 vs 400 vs 401 vs 403 vs 404 vs 429 vs 500
4. How do you design a clean REST endpoint?
5. Authentication vs authorization

### SQL
1. Find duplicate records.
2. Top spender per month.
3. INNER JOIN vs LEFT JOIN.
4. WHERE vs HAVING.
5. What is an index?
6. ROW_NUMBER vs RANK vs DENSE_RANK.

## Low-priority topics

Do **not** spend major prep time on:
- obscure Python internals
- deep asyncio implementation details
- advanced DSA unrelated to the role
- compiler/runtime trivia
- memorizing dozens of status codes
- learning entirely new frameworks just for the interview

## Answer style

For technical questions:
1. Give the direct answer first.
2. Explain the intuition.
3. Give one example from your own work.
4. Mention a limitation/trade-off when relevant.

For project questions:
**Problem → Design → What I implemented → Result → Challenge → What I learned / would improve**

## Japanese fluency prep

Prepare simple, natural answers for:
- 自己紹介
- 現在の仕事
- AIの経験
- 転職理由
- なぜProdaptか
- 日本でのキャリア
- LLMプロジェクトの簡単な説明

Do not memorize artificially advanced Japanese. Clarity and follow-up ability matter more.

## Tracking

- [ ] FastAPI/RAG project story ready
- [ ] Pythia/ATE project story ready
- [ ] Transformer project story ready
- [x] Async/await basic concept understood
- [ ] Python interview essentials
- [ ] REST/FastAPI discussion
- [ ] SQL practical set
- [ ] Transformer/LLM fundamentals
- [ ] RAG/enterprise AI
- [ ] Production/system design
- [ ] Japanese self-introduction
- [ ] Full Round 1 mock
- [ ] Full Round 2 mock
- [ ] Fluency mock

## External interview evidence

- Prodapt Python/Data Engineer interview report (2025): https://www.linkedin.com/posts/theujjawlkumar_dataengineering-pyspark-sql-activity-7392901518602981376-B_hs
- Prodapt DevOps interview report (2025): https://www.linkedin.com/posts/karthikss07_my-interview-experience-at-prodapt-solutions-activity-7350068357645148160-nCG5
- Prodapt Associate Software Engineer interview reports: https://www.ambitionbox.com/interviews/prodapt-solutions-interview-questions/associate-software-engineer
