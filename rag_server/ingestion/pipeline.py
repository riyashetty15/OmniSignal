"""
rag_server/ingestion/pipeline.py
=================================
IngestionPipeline — ingests raw documents (PDF / docx / txt) into PGVector.

Steps:
  1. Parse       — extract plain text from the uploaded file
  2. Chunk       — split with overlap to preserve context across boundaries
  3. Enrich      — attach LLM-extracted metadata (region, doc_type, date) to each chunk
  4. Embed       — generate embedding vector per chunk
  5. Store       — write chunk + embedding + metadata into the correct module table
"""

from __future__ import annotations
import io
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from retrieval.pipeline import HybridRetrievalPipeline, _get_embedding
from shared_config import MODULE_VECTOR_STORES


# ── Configuration ──────────────────────────────────────────────────────────────

CHUNK_SIZE    = 512    # target tokens per chunk (approximate by characters ÷ 4)
CHUNK_OVERLAP = 64     # overlap between consecutive chunks (in tokens)
CHUNK_CHARS   = CHUNK_SIZE * 4
OVERLAP_CHARS = CHUNK_OVERLAP * 4


# ── Text extraction ────────────────────────────────────────────────────────────

def _extract_text(content: bytes, filename: str) -> str:
    """
    Extracts plain text from PDF, docx, or plain text files.
    Requires: pypdf2 (PDF), python-docx (docx).
    Falls back gracefully if libraries are absent.
    """
    fname = filename.lower()

    if fname.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            return "\n\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )
        except ImportError:
            # pypdf not installed — try raw byte decode
            return content.decode("utf-8", errors="ignore")

    if fname.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(content))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            return content.decode("utf-8", errors="ignore")

    # txt / md / csv / json → decode directly
    return content.decode("utf-8", errors="ignore")


# ── Chunking ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    """
    Splits text into overlapping chunks.
    Splits on paragraph / sentence boundaries where possible.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())   # normalise whitespace

    # Split on paragraph boundaries first
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[str] = []
    current_chars = 0
    current_parts: list[str] = []

    for para in paragraphs:
        if current_chars + len(para) <= CHUNK_CHARS:
            current_parts.append(para)
            current_chars += len(para)
        else:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
            # Start new chunk with overlap from tail of previous
            overlap_text = "\n\n".join(current_parts)[-OVERLAP_CHARS:]
            current_parts = [overlap_text, para] if overlap_text else [para]
            current_chars = sum(len(p) for p in current_parts)

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return [c for c in chunks if c.strip()]


# ── Metadata enrichment ────────────────────────────────────────────────────────

_YEAR_RE   = re.compile(r"\b(20\d{2})\b")
_ZIP_RE    = re.compile(r"\b\d{5}\b")
_REGION_RE = re.compile(
    r"\b(pacific northwest|southeast|midwest|northeast|southwest|west coast|east coast)\b",
    re.I,
)


def _enrich_metadata(chunk: str, base_metadata: dict) -> dict:
    """
    Extracts lightweight metadata signals from the chunk text to augment base metadata.
    Used for metadata pre-filtering during retrieval.
    """
    meta = dict(base_metadata)

    # Extract year if not already present
    if "year" not in meta:
        years = _YEAR_RE.findall(chunk)
        if years:
            meta["year"] = int(years[0])

    # Extract ZIP codes
    zips = _ZIP_RE.findall(chunk)
    if zips and "zip_codes" not in meta:
        meta["zip_codes"] = zips[:5]   # store first 5 found

    # Extract region
    region_match = _REGION_RE.search(chunk)
    if region_match and "region" not in meta:
        meta["region"] = region_match.group(0).lower()

    return meta


# ── IngestionPipeline ──────────────────────────────────────────────────────────

@dataclass
class IngestionResult:
    status:        str
    chunks_stored: int
    filename:      str
    module:        str
    table:         str
    error:         str | None = None

    def to_dict(self) -> dict:
        return {
            "status":        self.status,
            "chunks_stored": self.chunks_stored,
            "filename":      self.filename,
            "module":        self.module,
            "table":         self.table,
            "error":         self.error,
        }


class IngestionPipeline:
    """
    Stateless pipeline — depends on the shared HybridRetrievalPipeline for storage.
    """

    def __init__(self, pipeline: HybridRetrievalPipeline) -> None:
        self._retrieval = pipeline

    async def ingest(
        self,
        content:  bytes,
        filename: str,
        module:   str,
        metadata: dict,
    ) -> dict:
        """
        Full ingestion pipeline for one document.

        metadata should include at minimum:
          doc_type, year, title, filename
        Optional: region, channel, validated (bool), campaign_id
        """
        table = MODULE_VECTOR_STORES.get(module)
        if not table:
            return IngestionResult(
                status="error", chunks_stored=0,
                filename=filename, module=module, table="",
                error=f"Unknown module: {module}",
            ).to_dict()

        # 1. Extract text
        try:
            text = _extract_text(content, filename)
        except Exception as exc:
            return IngestionResult(
                status="error", chunks_stored=0,
                filename=filename, module=module, table=table,
                error=f"Text extraction failed: {exc}",
            ).to_dict()

        if not text.strip():
            return IngestionResult(
                status="error", chunks_stored=0,
                filename=filename, module=module, table=table,
                error="No text extracted from document",
            ).to_dict()

        # 2. Chunk
        chunks = _chunk_text(text)

        # 3–5. Enrich → embed → store each chunk
        stored = 0
        errors = []
        for i, chunk in enumerate(chunks):
            try:
                enriched_meta = _enrich_metadata(chunk, metadata)
                enriched_meta["chunk_index"] = i
                enriched_meta["total_chunks"] = len(chunks)

                embedding = await _get_embedding(chunk)
                await self._retrieval.store_chunk(
                    table=table, content=chunk,
                    embedding=embedding, metadata=enriched_meta,
                )
                stored += 1
            except Exception as exc:
                errors.append(f"chunk {i}: {exc}")

        status = "success" if stored > 0 else "error"
        error  = "; ".join(errors) if errors else None

        return IngestionResult(
            status=status, chunks_stored=stored,
            filename=filename, module=module, table=table,
            error=error,
        ).to_dict()
