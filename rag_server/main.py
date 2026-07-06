"""
rag_server/main.py
===================
RAG FastAPI server — port 8081.
Handles document ingestion and hybrid retrieval.
The LLM server calls this over HTTP for every retrieval request.
"""

from __future__ import annotations
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retrieval.pipeline import HybridRetrievalPipeline
from rag_server.ingestion.pipeline import IngestionPipeline
from shared_config import MODULE_VECTOR_STORES, RAG_SERVER_PORT

PGVECTOR_CONN = os.getenv("PGVECTOR_CONN", "postgresql+asyncpg://fiber:fiber@localhost:5432/fiberorbit_db")
SQLITE_DB     = os.getenv("SQLITE_DB",     "data/context_memory.db")

retriever:  HybridRetrievalPipeline | None = None
ingestor:   IngestionPipeline | None       = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, ingestor
    retriever = HybridRetrievalPipeline(pgvector_conn=PGVECTOR_CONN, sqlite_db=SQLITE_DB)
    await retriever.initialize()
    ingestor  = IngestionPipeline(pipeline=retriever)
    yield
    if retriever:
        await retriever.close()


app = FastAPI(title="FiberOrbit RAG Server", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Models ────────────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query:    str
    module:   str
    filters:  dict  = {}
    top_k:    int   = 8
    strategy: str   = "hybrid"   # hybrid | metadata | vector


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "modules": list(MODULE_VECTOR_STORES.keys())}


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    if req.module not in MODULE_VECTOR_STORES and req.module != "multi":
        raise HTTPException(400, f"Unknown module: {req.module}")
    docs = await retriever.retrieve(
        query=req.query, module=req.module,
        filters=req.filters, top_k=req.top_k, strategy=req.strategy,
    )
    return {
        "docs":       [d.to_dict() for d in docs],
        "doc_count":  len(docs),
        "module":     req.module,
        "top_score":  docs[0].score if docs else 0.0,
    }


@app.post("/ingest")
async def ingest_document(
    file:      UploadFile = File(...),
    module:    str = Form(...),
    doc_type:  str = Form(...),
    year:      int = Form(...),
    title:     str = Form(default=""),
    validated: bool = Form(default=False),
    channel:   str = Form(default=""),
    region:    str = Form(default=""),
):
    """
    Ingest a PDF or docx into the correct module vector store.

    module:   campaign | fiber_network | financial | competitive | seo | hr
    doc_type: campaign_report | network_report | validated_baseline | hr_policy | ...
    """
    if module not in MODULE_VECTOR_STORES:
        raise HTTPException(400, f"Invalid module: {module}")

    content  = await file.read()
    metadata = {
        "doc_type": doc_type, "year": year, "validated": validated,
        "channel":  channel,  "region": region,
        "title":    title or file.filename,
        "filename": file.filename,
    }
    result = await ingestor.ingest(
        content=content, filename=file.filename,
        module=module, metadata=metadata,
    )
    return result


@app.get("/modules")
async def list_modules():
    return {"modules": list(MODULE_VECTOR_STORES.keys())}


if __name__ == "__main__":
    uvicorn.run("rag_server.main:app", host="0.0.0.0", port=RAG_SERVER_PORT, reload=True)