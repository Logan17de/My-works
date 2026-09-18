# Prodapt Tokyo AI SDE Interview Preparation Plan

**Interview:** 24 Sep 2026, 7:00 PM JST  
**Likely role:** Senior Deployed Engineer — SDE 2  
**Format confirmed by recruiter:** 3 rounds — 2 technical discussions in English + final fluency check  
**Preparation rule:** Prepare as an **AI engineer with strong model-level knowledge**, not as a research scientist. Optimize for spoken engineering reasoning, resume defense, applied AI, backend/API, SQL, system design, and customer-facing judgment.

---

## What Prodapt is likely testing

Prodapt's deployed-engineering model combines:

- Applied AI: LLMs, prompt engineering, RAG, agents
- Software engineering: backend, frontend, APIs, system design, DevOps/LLMOps
- Data engineering: SQL, pipelines, integration, data quality
- Consulting: requirements, solution design, customer interaction, measurable outcomes

The interview is therefore not mainly about knowing Transformer theory. The strongest signal is:

> I understand LLMs deeply **and** I can turn an ambiguous business problem into a measurable, secure, production-oriented AI system.

### Expected importance

| Area | Priority |
|---|---|
| Own projects / CV | ★★★★★ |
| LLM application engineering | ★★★★★ |
| Python | ★★★★★ |
| REST / FastAPI | ★★★★★ |
| RAG / retrieval | ★★★★★ |
| AI system design | ★★★★★ |
| SQL | ★★★★☆ |
| Evaluation / hallucination / safety | ★★★★☆ |
| Full-stack concepts | ★★★★☆ |
| Microservices | ★★★★☆ |
| Transformers / attention | ★★★★☆ |
| Customer / scenario questions | ★★★★☆ |
| Deployment / cloud / LLMOps | ★★★☆☆ |
| React details | ★★★☆☆ |
| Pure DSA coding | ★★☆☆☆ |

**Do not grind LeetCode.** The recruiter said these are discussion rounds.

---

# Core answer structure

## Technical questions
1. Direct answer
2. Intuition
3. Example from my own work
4. Trade-off / limitation

## Project questions
**Problem → My responsibility → Architecture → Difficult decision → Result → Failure/lesson → What I would improve**

## System-design questions
Start with:
- Who are the users?
- What problem are we solving?
- What is the cost of a wrong answer?
- What data exists?
- What scale / latency?
- What security/privacy requirements?
- How will success be measured?

Then design.

---

# Three project stories to make bulletproof

## 1. Pythia / Token MOD + ATE — strongest technical project

Use this to demonstrate:
- Transformer knowledge
- adaptation/fine-tuning
- deterministic experimentation
- causal masking
- evaluation
- overfitting
- debugging
- scientific integrity

Must answer from memory:
- Why Pythia?
- What was frozen?
- What was trainable?
- What was the baseline?
- Why full fine-tuning as a control?
- Why lock tokenizer/data/splits?
- What is perplexity?
- Why can't perplexity alone select a model?
- What caused sequential overfitting?
- How did you detect future-token leakage?
- What did you invalidate after finding leakage?
- What did the negative results teach you?

**Key interview story:** promising results were invalidated after future-token leakage was discovered. Explain why that changed your evaluation discipline.

## 2. Applied GenAI / RAG assistant — strongest role-fit project

Use this to demonstrate:
- Python
- FastAPI
- REST
- LLM APIs
- FAISS
- embeddings
- retrieval
- provider abstraction
- production thinking

Must answer:
- What user problem?
- End-to-end request flow?
- Why FastAPI?
- Why FAISS?
- How retrieval worked?
- How embeddings were generated?
- How prompt/context was assembled?
- How failures/timeouts were handled?
- RAG vs fine-tuning?
- How would you reduce hallucination?
- How would you evaluate retrieval separately from generation?
- What would change for production scale?

## 3. From-scratch Transformer / Gemma residual learners

Use this to prove you understand models beyond APIs.

Must answer:
- Q/K/V
- causal mask
- multi-head attention
- RoPE
- RMSNorm
- SwiGLU
- pre-norm
- untied LM head
- zero-effect initialization
- freezing previous learners
- parameter matching
- leakage-aware data generation

---

# High-probability Technical Round 1 questions

## Opening / CV

1. Tell me about yourself.
2. Explain your current GenAI work at TCS.
3. Tell me about one AI/LLM project you built end-to-end.
4. What exactly did you implement yourself?
5. What was your hardest technical problem?
6. Tell me about a failure or invalid result.
7. How do you evaluate whether your experiment actually improved something?
8. Your independent research is unusual — is it production experience?
9. Why Mechanical Engineering if you're now an AI engineer?
10. What is your strongest project?

## Transformer / LLM fundamentals

11. Explain a Transformer architecture.
12. Explain self-attention and Q/K/V.
13. Why divide attention logits by sqrt(d_k)?
14. What is multi-head attention?
15. What is a causal mask?
16. How did future-token leakage affect your previous experiment?
17. How did you detect the leakage?
18. What is perplexity?
19. Can perplexity be compared across different tokenizers?
20. Pretraining vs SFT vs fine-tuning.
21. Full fine-tuning vs LoRA / PEFT.
22. What causes catastrophic forgetting?
23. What is RoPE?
24. What is RMSNorm?
25. What is SwiGLU?
26. What does pre-norm mean?

## Applied GenAI / RAG

27. Fine-tuning vs RAG vs prompt engineering.
28. When would you **not** fine-tune?
29. Explain a RAG pipeline end-to-end.
30. How do you choose chunk size?
31. What is an embedding?
32. What is FAISS?
33. Cosine similarity vs L2 distance?
34. What if retrieval returns irrelevant chunks?
35. Why can RAG hallucinate even when retrieval is correct?
36. How would you evaluate a RAG system?
37. How would you reduce hallucination?
38. What is prompt engineering?
39. Zero-shot vs few-shot.
40. What is an AI agent?
41. Agent vs deterministic workflow?
42. What is tool/function calling?
43. How would you stop an AI agent from doing dangerous actions?

---

# Python / FastAPI / REST topics

These matter because they are closer to the SDE2 gap than your Transformer knowledge.

## Python questions

44. List vs tuple vs set vs dictionary.
45. Mutable vs immutable.
46. Shallow copy vs deep copy.
47. Generator vs normal function.
48. What is a decorator?
49. Give a practical decorator example.
50. Explain OOP using a real project example.
51. Exception handling — how would you handle an upstream model failure?
52. Context manager — what problem does it solve?
53. async/await — when is it useful?
54. async vs threads vs processes.
55. Why doesn't async automatically make local GPU inference faster?

**Already covered:** async/await basic mental model.

## FastAPI

56. Why FastAPI?
57. How does request validation work?
58. What does Pydantic do?
59. How would you structure an LLM endpoint?
60. How would you authenticate it?
61. How would you handle validation errors?
62. How would you handle an LLM request that takes 30 seconds?
63. How would you stream an LLM response?
64. How would you implement timeout / retry / cancellation?
65. How would you rate-limit the endpoint?

## REST

66. GET vs POST vs PUT vs PATCH vs DELETE.
67. What does idempotent mean?
68. Which methods are normally idempotent?
69. Authentication vs authorization.
70. Explain 200 / 201 / 204 / 400 / 401 / 403 / 404 / 409 / 422 / 429 / 500 / 503.
71. Design a clean REST endpoint for an LLM conversation.

---

# SQL and microservices — key weakness-remediation area

## SQL

72. INNER JOIN vs LEFT JOIN.
73. Why would you use LEFT JOIN?
74. What is an index?
75. Should every column be indexed?
76. How do you optimize a slow SQL query?
77. What is a transaction?
78. Explain ACID.
79. SQL vs vector database.
80. Find duplicate rows.
81. Top spender per month.
82. WHERE vs HAVING.
83. ROW_NUMBER vs RANK vs DENSE_RANK.
84. Primary key vs foreign key.
85. What is normalization?

## Microservices

86. What is a microservice?
87. Microservices vs modular monolith.
88. How do services communicate?
89. REST vs asynchronous messaging.
90. What if Service B fails after Service A writes its DB?
91. What is idempotency in distributed systems?
92. Why use a queue?
93. What is a circuit breaker?
94. What would you monitor between services?

---

# Technical Round 2 — scenario / system design

## Highest-priority design

### Design an enterprise RAG chatbot

Start with requirements:
- internal or customer-facing?
- document types?
- update frequency?
- permission-sensitive?
- required accuracy?
- number of users?
- cloud/on-prem?
- can customer data leave environment?

Then cover:

```text
Documents
  ↓
Parsing / cleaning
  ↓
Chunking
  ↓
Metadata + ACL extraction
  ↓
Embeddings
  ↓
Vector index + relational metadata
  ↓
User query
  ↓
Authentication / authorization
  ↓
Query transformation
  ↓
Retrieval
  ↓
Permission filtering
  ↓
Reranking
  ↓
Prompt assembly
  ↓
LLM
  ↓
Output validation
  ↓
Answer + citations
  ↓
Logging / feedback / evaluation
```

Then discuss:
- reliability
- security
- latency
- cost
- observability
- deployment
- evaluation
- rollback

## Scenario questions

95. A customer says, “Build us an AI assistant.” What do you do first?
96. Customer wants 100% accuracy. How do you respond?
97. RAG gives wrong answers. How do you debug it?
98. Latency went from 3s to 15s. How do you investigate?
99. LLM provider is unavailable. What happens?
100. How do you deploy a new model safely?
101. How do you monitor an LLM application?
102. Large model vs small model — how do you choose?
103. How do you evaluate an LLM before production?
104. What do you do about prompt injection?
105. How do you protect customer data with an external LLM API?
106. Customer wants an autonomous agent connected to production databases. What concerns you?
107. Requirement changes halfway through implementation. What do you do?
108. Client disagrees with your technical recommendation. What do you do?
109. Your AI service works for 5 users but slows badly at 500. What do you investigate?
110. How would you build a multi-tenant GenAI API?

---

# Japanese / fluency round

Prepare simple, natural answers for:

1. 自己紹介をお願いします。
2. 現在の仕事について説明してください。
3. AIの経験について教えてください。
4. なぜ転職したいですか。
5. なぜProdaptに興味がありますか。
6. 日本でどのようなキャリアを作りたいですか。
7. LLMのプロジェクトを簡単に説明してください。
8. 日本語で仕事をすることはできますか。

Useful recovery phrases:

- すみません、もう一度お願いします。
- もう少しゆっくり話していただけますか。
- すみません、その言葉の意味を確認してもいいですか。

Do not pretend to have business-level Japanese if that is not accurate. Simple, clear, honest communication is better.

---

# Revised day-by-day plan

## Sep 18–19 — Make the CV bulletproof

For **every technical phrase on the resume**, answer:

- What is it?
- Why did I use it?
- How did I implement it?
- What alternative existed?
- What failed?
- What metric did I use?
- What would I change today?

Priority terms:
Pythia, Gemma 3, attention, RoPE, RMSNorm, SwiGLU, FAISS, UltraChat, leakage detection, checkpoint migration, FastAPI, WebSockets, OpenAI/Gemini APIs, React, deterministic datasets, perplexity, RAG, agents.

### Today's concrete session
1. Tell me about yourself.
2. Explain current GenAI work at TCS.
3. Explain Pythia/ATE in 2 minutes.
4. Explain leakage bug.
5. Explain FastAPI/RAG project end-to-end.
6. RAG vs fine-tuning vs prompting.
7. Explain RAG pipeline.
8. How do you evaluate RAG?
9. Why FastAPI?
10. Build one LLM REST endpoint verbally.

Only fill Python gaps when a question depends on them.

## Sep 20 — Python + FastAPI + REST
- data structures
- OOP
- exceptions
- context managers
- decorators
- generators
- async/await
- threads vs processes
- FastAPI routing
- Pydantic validation
- authentication
- REST methods
- status codes
- idempotency
- pagination
- rate limiting
- timeout/retry
- streaming

## Sep 21 — SQL + microservices
Highest-priority weakness-remediation day.

SQL:
- SELECT / WHERE / GROUP BY / HAVING
- INNER / LEFT JOIN
- CTEs / subqueries
- indexes
- transactions / ACID
- normalization
- primary / foreign keys
- query plans
- window functions

Microservices:
- service boundaries
- REST vs events
- queues
- retries
- idempotency
- circuit breakers
- distributed consistency
- caching
- observability

## Sep 22 — LLM application architecture
Practice out loud:
- enterprise RAG chatbot
- customer-support AI assistant
- document summarization platform
- AI coding assistant with tools
- multi-tenant GenAI API

For every design cover:
**data → API → model → DB → retrieval → security → evaluation → monitoring → scaling → cost**

## Sep 23 — Two full technical mocks

### Round A — 60 min
- introduction
- current work
- Transformer
- attention
- causal mask
- leakage
- fine-tuning
- RAG
- hallucination
- Pythia
- Python async
- FastAPI
- SQL JOIN
- microservices

### Round B — 60 min
- design enterprise RAG
- debug poor retrieval
- LLM outage
- latency
- customer data
- AI evaluation
- deploy model update
- changing requirement
- failed project
- why Prodapt

Then final Japanese fluency mock.

## Sep 24 — interview day
Only light review:
- opening
- strongest two projects
- RAG system-design structure
- SQL fundamentals
- Japanese self-introduction
- no large new topics

---

# Top 20 questions to master first

1. Tell me about yourself.
2. Explain your current GenAI work at TCS.
3. Explain one LLM project end-to-end.
4. Explain Transformer architecture.
5. Explain self-attention and Q/K/V.
6. What is a causal mask?
7. Explain the leakage bug you discovered.
8. Fine-tuning vs RAG vs prompting.
9. Explain a RAG pipeline.
10. How do you reduce hallucination?
11. How do you evaluate an LLM/RAG system?
12. Explain Python async/await.
13. How would you create an LLM API using FastAPI?
14. INNER JOIN vs LEFT JOIN.
15. What is an index and when does it help?
16. What is a microservice and what are its trade-offs?
17. Design an enterprise AI chatbot.
18. Your AI service became slow — how do you debug it?
19. Customer requirement is ambiguous — what do you do?
20. Why Prodapt / why this role?

---

# Tracking

- [ ] 60-second introduction
- [ ] Current TCS GenAI answer
- [ ] Pythia/ATE 2-minute answer
- [ ] Leakage/debugging answer
- [ ] FastAPI/RAG end-to-end answer
- [ ] Transformer/attention explanation
- [ ] Fine-tuning vs RAG vs prompting
- [ ] RAG evaluation answer
- [x] Async/await basic concept
- [ ] Python/FastAPI/REST day
- [ ] SQL/microservices day
- [ ] LLM architecture day
- [ ] Japanese self-introduction
- [ ] Full Technical Round A
- [ ] Full Technical Round B
- [ ] Fluency mock

---

# Interview reasoning phrases

Use these naturally:

- “First I would clarify the requirement…”
- “There are two trade-offs here…”
- “I would separate retrieval failure from generation failure…”
- “I would measure this rather than assume…”
- “For a prototype I might choose X, but for production I would add Y…”
- “The answer depends on latency, cost, reliability and security requirements…”
- “I would establish a baseline first…”
- “I would validate that with an evaluation set…”

The goal is not to sound memorized. The goal is to sound like an engineer who can reason under ambiguity.
