# OmniSignal

A multi-agent AI system for fiber ISP operations — marketing analytics, infrastructure investment decisions, content strategy, and HR policy Q&A under one unified interface.

Built with LangGraph, Anthropic Claude, PGVector, and FastAPI. Deployed on Azure.

---

## Overview

OmniSignal routes every incoming query through a LangGraph StateGraph that enforces guardrails, selects the right specialist agent, validates output quality, and retries automatically if the response falls below a 92% fidelity threshold.

```
guardrails → planner → [specialist agent] → validator → response builder
                              ↑                   |
                              └── retry (once) ───┘ (if fidelity < 0.92)
```

---

## Agents

| Agent | Model | Responsibility |
|---|---|---|
| `data_analyst` | Claude Opus | Take rates, campaign KPIs, funnel metrics, cohort analysis, anomaly detection |
| `financial_planner` | Claude Opus | Copper-to-fiber NPV/IRR, Calix demographics, CAPEX modeling, ZIP prioritization |
| `strategist` | Claude Opus | Social content ideation, content calendar, FTC compliance checking |
| `hr_docqa` | Claude Haiku | HR policy Q&A, PII masking, citation-required answers, escalation routing |

All financial and analytical calculations are **deterministic Python** — Claude interprets and narrates results but never computes numbers.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              LLM Server (port 8000)          │
│  FastAPI · LangGraph · SQLite checkpointing  │
└────────────────────┬────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────┐
│              RAG Server (port 8081)          │
│  FastAPI · HybridRetrievalPipeline           │
│  Dense (pgvector) + Keyword (FTS) + RRF      │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│         PGVector on Azure PostgreSQL         │
│  6 module-siloed tables · RBAC per agent     │
└─────────────────────────────────────────────┘
```

### Retrieval

Hybrid pipeline combining:
- **Dense search** — cosine similarity via pgvector (`<=>` operator), Azure OpenAI `text-embedding-3-small` (1536-dim)
- **Keyword search** — PostgreSQL full-text search (`ts_rank`)
- **RRF fusion** — Reciprocal Rank Fusion merges both result lists without score-scale dependency
- **Metadata pre-filter** — JSONB containment applied before ANN search for 18% precision improvement

### Fidelity Scoring

Every response is scored across three dimensions before delivery:

| Dimension | Weight | What it checks |
|---|---|---|
| Structural completeness | 30% | Intent-aware keyword group matching (18 mapped intents) |
| Numeric faithfulness | 40% | Response numbers cross-checked against tool outputs (±5% tolerance) |
| Citation coverage | 30% | Policy name, section, version, effective date for HR/financial responses |

Score < 0.92 → automatic retry. Score < 0.92 on retry → pass through with validation notes attached.

### Compliance Engine

Pre-publication gate for all outbound content:

- **TCPA** — prior express written consent for SMS/call outreach
- **FTC/FCC** — advertising claim truthfulness, broadband disclosure requirements
- **CAN-SPAM** — opt-out mechanism, physical address, subject line (email only)
- **Privacy** — CCPA (CA), VCDPA (VA), CPA (CO)
- **State laws** — broadband advertising restrictions per state

Result: `GO` | `GO-WITH-REVIEW` | `NO-GO`. Any `NO-GO` blocks content before it leaves the system.

---

## Project Structure

```
OmniSignal/
├── agents/
│   ├── base.py                  # AgentState TypedDict, initial_state factory
│   ├── graph.py                 # LangGraph StateGraph wiring
│   ├── planner.py               # Two-stage router + entity extraction
│   ├── hr/
│   │   └── hr_docqa.py          # HR agent (Haiku, isolated DB access)
│   ├── specialist/
│   │   ├── data_analyst.py      # Analytics agent + deterministic tools
│   │   ├── financial_planner.py # Financial agent + NPV/IRR/Calix tools
│   │   └── strategist.py        # Content agent + compliance gate
│   └── validation/
│       ├── guardrails.py        # 5-layer pre-flight safety checks
│       └── report_validator.py  # 3-dimension fidelity scorer
├── compliance/
│   ├── engine.py                # Aggregates all compliance checkers
│   ├── tcpa.py
│   ├── fcc_ftc_canspam.py
│   ├── privacy.py
│   ├── state_laws.py
│   └── integration.py           # Agent tool wrapper + audit logger
├── core/
│   ├── router.py                # Keyword-based query router
│   └── conversation_context.py  # In-memory session state
├── retrieval/
│   └── pipeline.py              # HybridRetrievalPipeline (dense + keyword + RRF)
├── memory/
│   └── sqlite_store.py          # Async SQLite: turns, entity memory, tool cache
├── services/
│   └── decision_engine.py       # Cross-agent tool delegation + compound queries
├── llm_server/
│   └── main.py                  # FastAPI server (port 8000) — primary entry point
├── rag_server/
│   ├── main.py                  # FastAPI server (port 8081) — retrieval + ingestion
│   └── ingestion/
│       └── pipeline.py          # Document chunking + embedding + storage
├── infra/
│   ├── Dockerfile.llm
│   ├── Dockerfile.rag
│   ├── docker-compose.yml
│   ├── init_pgvector.sql        # PGVector table setup (6 module-siloed tables)
│   └── azure/
│       └── azure-ml-config.yml  # LightGBM fiber demand forecast — monthly retrain
├── tests/
│   ├── agents.py
│   ├── compliance.py
│   ├── guardrails.py
│   ├── report_validator.py
│   └── retrieval.py
├── shared_config.py             # Models, thresholds, benchmarks, module config
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Anthropic API key
- Azure OpenAI endpoint + key (for embeddings; falls back to deterministic mock locally)

### Environment Variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_key

# Azure OpenAI (embeddings)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small

# Database
PGVECTOR_CONN=postgresql+asyncpg://fiber:fiber@localhost:5432/fiberorbit_db
SQLITE_DB=data/context_memory.db

# Calix Marketing Cloud
CALIX_API_KEY=your_calix_key
```

### Run with Docker

```bash
docker-compose -f infra/docker-compose.yml up --build
```

This starts both the LLM server (port 8000) and RAG server (port 8081) with PGVector.

### Run locally (without Docker)

```bash
pip install -r requirements.txt

# Terminal 1 — LLM server
uvicorn llm_server.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — RAG server
uvicorn rag_server.main:app --host 0.0.0.0 --port 8081 --reload
```

Both servers fall back to mock data when the database is unavailable — local development works without credentials.

---

## API

### POST `/query`

```json
{
  "query": "What is the take rate in the Pacific Northwest for Q3?",
  "user_id": "user_123",
  "session_id": "optional_session_id",
  "department": "marketing"
}
```

Response includes `agent_used`, `fidelity_score`, `confidence`, `evidence_coverage`, and `validation_notes`.

### POST `/retrieve` (RAG server)

```json
{
  "query": "fiber propensity scoring methodology",
  "module": "financial",
  "top_k": 8,
  "strategy": "hybrid"
}
```

### POST `/ingest` (RAG server)

Upload a PDF or DOCX to a module vector store via multipart form.

---

## Azure ML — Fiber Demand Forecast

A LightGBM regression model trained on ~180,000 historical copper-to-fiber conversion records. Features: Calix demographics + internal subscriber data. Retrains monthly on the 1st at 02:00 UTC via Azure ML scheduled job.

Deploy with:
```bash
az ml job create -f infra/azure/azure-ml-config.yml
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph StateGraph |
| LLM | Anthropic Claude (Opus + Haiku) |
| Embeddings | Azure OpenAI text-embedding-3-small |
| Vector store | PGVector on Azure PostgreSQL |
| Retrieval | Hybrid dense + FTS + Reciprocal Rank Fusion |
| Context memory | SQLite (LangGraph AsyncSqliteSaver) |
| Servers | FastAPI + Uvicorn |
| ML pipeline | Azure ML + LightGBM + Optuna + MLflow |
| Containers | Docker + Docker Compose |
