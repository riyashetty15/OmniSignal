"""
llm_server/main.py
===================
LLM FastAPI server — port 8000. THE PRIMARY ENTRY POINT.
Receives queries, manages sessions, runs the LangGraph graph,
persists context to SQLite, returns structured responses.

Start with:
  uvicorn llm_server.main:app --host 0.0.0.0 --port 8000 --workers 2
"""

from __future__ import annotations
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.graph import create_compiled_graph
from agents.base import initial_state
from core.router import route_query
from core.conversation_context import ConversationContext
from memory.sqlite_store import SQLiteContextStore
from services.decision_engine import plan_compound_query, execute_compound_query
import structlog

logger = structlog.get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
SQLITE_DB = os.getenv("SQLITE_DB", "data/context_memory.db")
SESSION_TTL_HOURS = 24

# ── App state ─────────────────────────────────────────────────────────────────
sessions: dict[str, dict]      = {}
context_store: SQLiteContextStore | None = None
compiled_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global context_store, compiled_graph
    import os
    os.makedirs("data", exist_ok=True)
    context_store = SQLiteContextStore(SQLITE_DB)
    await context_store.initialize()
    compiled_graph = await create_compiled_graph(sqlite_db=SQLITE_DB)
    logger.info("llm_server.ready")
    yield
    logger.info("llm_server.shutdown")


app = FastAPI(title="FiberOrbit LLM Server", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Pydantic models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:       str
    user_id:     str
    session_id:  str | None   = None
    department:  str          = "marketing"   # "hr" | "marketing" | "finance"
    stream:      bool         = False


class QueryResponse(BaseModel):
    session_id:       str
    query:            str
    response:         str
    agent_used:       str
    fidelity_score:   float
    confidence:       float
    evidence_coverage:str
    latency_ms:       float
    timestamp:        str
    validation_notes: list[str] = Field(default_factory=list)


# ── Session helpers ───────────────────────────────────────────────────────────

async def get_or_create_session(session_id: str | None, user_id: str, department: str) -> tuple[str, ConversationContext]:
    if session_id and session_id in sessions:
        sess = sessions[session_id]
        if datetime.utcnow() - sess["created_at"] < timedelta(hours=SESSION_TTL_HOURS):
            return session_id, sess["context"]
        del sessions[session_id]

    new_id = session_id or str(uuid.uuid4())
    history = await context_store.get_context(user_id, new_id, limit=10)
    entities = await context_store.get_entities(user_id)

    ctx = ConversationContext(
        session_id=new_id, user_id=user_id, department=department,
        history=[{"role":"assistant","content":h["response"],"metadata":{"agent":h["agent"]}}
                 if i%2==1 else {"role":"user","content":h["query"]}
                 for i,h in enumerate(history)],
        entity_memory=entities,
    )
    sessions[new_id] = {"context": ctx, "created_at": datetime.utcnow()}
    return new_id, ctx


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "FiberOrbit LLM Server", "sessions": len(sessions)}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    import time
    start = time.perf_counter()

    session_id, ctx = await get_or_create_session(req.session_id, req.user_id, req.department)

    # Semantic routing
    route = route_query(req.query, ctx.get_llm_messages())

    # ── Compound query check: if the query spans multiple agents, bypass the
    #    single-agent graph and run the DecisionEngine compound path instead.
    compound_plan = plan_compound_query(req.query)
    if compound_plan:
        logger.info("compound_query.detected", description=compound_plan.description)
        compound_result = await execute_compound_query(
            plan=compound_plan,
            context=await context_store.get_context(req.user_id, session_id, limit=10),
        )
        latency_ms = (time.perf_counter() - start) * 1000
        final_response = compound_result.merged_response
        agent_used = "+".join(s["agent"] for s in compound_plan.steps)

        await context_store.save_turn(
            user_id=req.user_id, session_id=session_id,
            query=req.query, response=final_response,
            agent=agent_used,
            metadata={"compound": True, "steps": compound_result.steps_completed},
        )
        ctx.add_turn("user",      req.query)
        ctx.add_turn("assistant", final_response[:500], {"agent": agent_used})

        logger.info("compound_query.complete", session_id=session_id,
                    agents=agent_used, latency_ms=f"{latency_ms:.0f}",
                    steps=compound_result.steps_completed)

        return QueryResponse(
            session_id        = session_id,
            query             = req.query,
            response          = final_response,
            agent_used        = agent_used,
            fidelity_score    = 0.0,
            confidence        = 0.0,
            evidence_coverage = "compound",
            latency_ms        = round(latency_ms, 1),
            timestamp         = datetime.utcnow().isoformat(),
            validation_notes  = [f"Compound query: {compound_plan.description}"],
        )

    # Build initial AgentState
    context_window = await context_store.get_context(req.user_id, session_id, limit=10)
    entity_memory  = await context_store.get_entities(req.user_id)

    state = initial_state(
        session_id    = session_id,
        user_id       = req.user_id,
        user_query    = req.query,
        department    = req.department,
        context_window= context_window,
    )
    state["routed_agent"]  = route.agent
    state["entity_memory"] = entity_memory

    # Execute the LangGraph
    config      = {"configurable": {"thread_id": session_id}}
    final_state = await compiled_graph.ainvoke(state, config=config)

    latency_ms = (time.perf_counter() - start) * 1000

    # Persist to SQLite
    final_response = final_state.get("final_response") or "No response generated."
    await context_store.save_turn(
        user_id=req.user_id, session_id=session_id,
        query=req.query, response=final_response,
        agent=route.agent,
        metadata={"fidelity_score": final_state.get("fidelity_score",0)},
    )

    # Update session context
    ctx.add_turn("user",      req.query)
    ctx.add_turn("assistant", final_response[:500], {"agent": route.agent})

    logger.info("query.complete", session_id=session_id,
                agent=route.agent, latency_ms=f"{latency_ms:.0f}",
                fidelity=final_state.get("fidelity_score",0))

    return QueryResponse(
        session_id       = session_id,
        query            = req.query,
        response         = final_response,
        agent_used       = route.agent,
        fidelity_score   = final_state.get("fidelity_score", 0.0),
        confidence       = final_state.get("output_confidence", 0.0),
        evidence_coverage= final_state.get("evidence_coverage", "unknown"),
        latency_ms       = round(latency_ms, 1),
        timestamp        = datetime.utcnow().isoformat(),
        validation_notes = final_state.get("validation_notes", []),
    )


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    """Streaming endpoint — yields agent progress events in real time."""
    await ws.accept()
    try:
        data = await ws.receive_json()
        req  = QueryRequest(**data, session_id=session_id)
        _, ctx = await get_or_create_session(session_id, req.user_id, req.department)
        route  = route_query(req.query, ctx.get_llm_messages())
        state  = initial_state(session_id=session_id, user_id=req.user_id,
                               user_query=req.query, department=req.department)
        state["routed_agent"] = route.agent
        config = {"configurable": {"thread_id": session_id}}

        import json
        async for event in compiled_graph.astream_events(state, config=config, version="v2"):
            etype = event.get("event","")
            if etype == "on_chain_start":
                await ws.send_text(json.dumps({"type":"agent_start","agent":event.get("name","")}))
            elif etype == "on_chain_end":
                await ws.send_text(json.dumps({"type":"agent_done","agent":event.get("name","")}))
            elif etype == "on_llm_stream":
                chunk = event.get("data",{}).get("chunk",{})
                if hasattr(chunk,"content") and chunk.content:
                    await ws.send_text(json.dumps({"type":"token","content":chunk.content}))
        await ws.send_text(json.dumps({"type":"done"}))
    except WebSocketDisconnect:
        pass


@app.delete("/session/{session_id}")
async def close_session(session_id: str):
    sessions.pop(session_id, None)
    return {"closed": session_id}


@app.get("/status")
async def status():
    total_sessions = len(sessions)
    return {
        "active_sessions": total_sessions,
        "sqlite_db":       SQLITE_DB,
        "timestamp":       datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run("llm_server.main:app", host="0.0.0.0", port=8000, reload=True)