"""
tests/retrieval.py
===================
Tests for the HybridRetrievalPipeline and IngestionPipeline.
These tests run entirely without a live database — the pipeline falls back
to mock results when no DB pool is available.
Run with: pytest tests/retrieval.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from retrieval.pipeline import (
    HybridRetrievalPipeline, RetrievedDocument,
    _rrf_fuse, _build_where, _mock_results,
)
from rag_server.ingestion.pipeline import IngestionPipeline, _chunk_text, _extract_text, _enrich_metadata


# ─────────────────────────────────────────────────────────────────────────────
#  RetrievedDocument
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievedDocument:
    def test_to_dict_has_required_keys(self):
        doc = RetrievedDocument(
            doc_id="d1", content="Test content", metadata={"module": "hr"},
            score=0.85, source="vector",
        )
        d = doc.to_dict()
        assert all(k in d for k in ["doc_id", "content", "metadata", "score", "source"])

    def test_score_preserved(self):
        doc = RetrievedDocument("d1", "c", {}, 0.923456, "hybrid")
        assert doc.to_dict()["score"] == 0.923456


# ─────────────────────────────────────────────────────────────────────────────
#  RRF fusion
# ─────────────────────────────────────────────────────────────────────────────

class TestRRFFusion:
    def _make_docs(self, ids: list[str], source: str) -> list[RetrievedDocument]:
        return [RetrievedDocument(did, f"content {did}", {}, 0.9 - i * 0.1, source)
                for i, did in enumerate(ids)]

    def test_rrf_deduplicates(self):
        dense   = self._make_docs(["a", "b", "c"], "vector")
        keyword = self._make_docs(["b", "c", "d"], "keyword")
        fused   = _rrf_fuse(dense, keyword)
        ids = [d.doc_id for d in fused]
        assert len(ids) == len(set(ids))   # no duplicates

    def test_rrf_boosts_docs_in_both_lists(self):
        dense   = self._make_docs(["a", "b"], "vector")
        keyword = self._make_docs(["b", "c"], "keyword")
        fused   = _rrf_fuse(dense, keyword)
        # "b" appears in both — should rank higher than "a" or "c" alone
        fused_ids = [d.doc_id for d in fused]
        assert fused_ids.index("b") < fused_ids.index("a") or \
               fused_ids.index("b") < fused_ids.index("c")

    def test_rrf_sorted_descending(self):
        dense   = self._make_docs(["x", "y", "z"], "vector")
        keyword = self._make_docs(["p", "q"], "keyword")
        fused   = _rrf_fuse(dense, keyword)
        scores  = [d.score for d in fused]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_empty_inputs(self):
        assert _rrf_fuse([], []) == []

    def test_rrf_one_empty(self):
        dense = [RetrievedDocument("a", "c", {}, 0.9, "vector")]
        fused = _rrf_fuse(dense, [])
        assert len(fused) == 1
        assert fused[0].doc_id == "a"

    def test_rrf_source_set_to_hybrid(self):
        dense   = [RetrievedDocument("a", "c", {}, 0.9, "vector")]
        keyword = [RetrievedDocument("b", "c", {}, 0.8, "keyword")]
        fused   = _rrf_fuse(dense, keyword)
        assert all(d.source == "hybrid" for d in fused)


# ─────────────────────────────────────────────────────────────────────────────
#  WHERE clause builder
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildWhere:
    def test_empty_filters(self):
        clause, params = _build_where({}, start_idx=3)
        assert clause == ""
        assert params == []

    def test_single_filter(self):
        clause, params = _build_where({"doc_type": "policy"}, start_idx=3)
        assert "$3" in clause
        assert len(params) == 1
        assert '"doc_type"' in params[0] or "doc_type" in params[0]

    def test_multiple_filters_increment_indices(self):
        clause, params = _build_where(
            {"doc_type": "report", "year": 2024}, start_idx=3
        )
        assert "$3" in clause
        assert "$4" in clause
        assert len(params) == 2


# ─────────────────────────────────────────────────────────────────────────────
#  Mock results fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestMockResults:
    def test_returns_correct_count(self):
        results = _mock_results("test query", top_k=3)
        assert len(results) == 3

    def test_scores_descending(self):
        results = _mock_results("query", top_k=3)
        scores  = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_source_is_mock(self):
        results = _mock_results("q", top_k=2)
        assert all(r.source == "mock" for r in results)


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline (no-DB fallback path)
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridRetrievalPipeline:
    @pytest.mark.asyncio
    async def test_initialize_without_db_does_not_raise(self):
        pipeline = HybridRetrievalPipeline(
            pgvector_conn="postgresql://invalid:5432/nodb",
            sqlite_db="data/test.db",
        )
        # Should not raise — just set pool to None
        await pipeline.initialize()
        assert pipeline._pool is None

    @pytest.mark.asyncio
    async def test_retrieve_without_db_returns_mocks(self):
        pipeline = HybridRetrievalPipeline("postgresql://invalid/nodb", "test.db")
        await pipeline.initialize()
        results = await pipeline.retrieve("fiber take rate", module="campaign")
        assert len(results) > 0
        assert all(isinstance(r, RetrievedDocument) for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_unknown_module_returns_mocks(self):
        pipeline = HybridRetrievalPipeline("postgresql://invalid/nodb", "test.db")
        await pipeline.initialize()
        results = await pipeline.retrieve("query", module="nonexistent_module")
        assert len(results) > 0   # falls back to mock


# ─────────────────────────────────────────────────────────────────────────────
#  Chunking
# ─────────────────────────────────────────────────────────────────────────────

class TestChunking:
    def test_short_text_single_chunk(self):
        chunks = _chunk_text("Short paragraph about fiber.")
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        long_text = ("This is a paragraph about fiber broadband. " * 30 + "\n\n") * 10
        chunks = _chunk_text(long_text)
        assert len(chunks) > 1

    def test_chunks_are_non_empty(self):
        chunks = _chunk_text("Para one.\n\nPara two.\n\nPara three.")
        assert all(len(c.strip()) > 0 for c in chunks)

    def test_chunks_cover_all_content(self):
        text = "unique_word_alpha\n\n" + ("filler " * 200 + "\n\n") * 5 + "unique_word_beta"
        chunks = _chunk_text(text)
        full  = " ".join(chunks)
        assert "unique_word_alpha" in full
        assert "unique_word_beta" in full


# ─────────────────────────────────────────────────────────────────────────────
#  Text extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestTextExtraction:
    def test_plain_text_extraction(self):
        content = b"Hello world fiber broadband"
        text = _extract_text(content, "document.txt")
        assert "Hello world" in text

    def test_utf8_extraction(self):
        content = "Fiber speeds of 1 Gbps available now.".encode("utf-8")
        text = _extract_text(content, "notice.txt")
        assert "Gbps" in text


# ─────────────────────────────────────────────────────────────────────────────
#  Metadata enrichment
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataEnrichment:
    def test_year_extracted(self):
        meta = _enrich_metadata("This report covers Q3 2024 performance.", {"doc_type": "report"})
        assert meta["year"] == 2024

    def test_zip_extracted(self):
        meta = _enrich_metadata("Coverage in ZIP 98101 and 90210.", {})
        assert "98101" in meta["zip_codes"]
        assert "90210" in meta["zip_codes"]

    def test_region_extracted(self):
        meta = _enrich_metadata("Fiber rollout in the Pacific Northwest region.", {})
        assert meta["region"] == "pacific northwest"

    def test_base_metadata_preserved(self):
        base = {"doc_type": "policy", "year": 2023, "module": "hr"}
        meta = _enrich_metadata("Some content here.", base)
        assert meta["doc_type"] == "policy"
        assert meta["module"]   == "hr"

    def test_base_year_not_overwritten(self):
        base = {"year": 2022}
        meta = _enrich_metadata("Report for 2024.", base)
        assert meta["year"] == 2022   # base takes priority


# ─────────────────────────────────────────────────────────────────────────────
#  IngestionPipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestionPipeline:
    @pytest.mark.asyncio
    async def test_ingest_unknown_module_returns_error(self):
        mock_pipeline = MagicMock()
        ingestor = IngestionPipeline(pipeline=mock_pipeline)
        result = await ingestor.ingest(
            content=b"some text", filename="file.txt",
            module="nonexistent", metadata={},
        )
        assert result["status"] == "error"
        assert "Unknown module" in result["error"]

    @pytest.mark.asyncio
    async def test_ingest_empty_content_returns_error(self):
        mock_pipeline = MagicMock()
        ingestor = IngestionPipeline(pipeline=mock_pipeline)
        result = await ingestor.ingest(
            content=b"   ", filename="empty.txt",
            module="hr", metadata={"doc_type": "policy", "year": 2024, "title": "T"},
        )
        assert result["status"] == "error"
        assert "No text" in result["error"]

    @pytest.mark.asyncio
    async def test_ingest_valid_text_calls_store(self):
        mock_pipeline = MagicMock()
        mock_pipeline.store_chunk = AsyncMock(return_value="chunk_id_1")
        ingestor = IngestionPipeline(pipeline=mock_pipeline)

        with patch("rag_server.ingestion.pipeline._get_embedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [0.1] * 1536
            result = await ingestor.ingest(
                content  = b"This is a valid document about fiber broadband.",
                filename = "fiber_policy.txt",
                module   = "hr",
                metadata = {"doc_type": "policy", "year": 2024, "title": "Fiber Policy"},
            )

        assert result["status"] == "success"
        assert result["chunks_stored"] >= 1
        assert result["module"] == "hr"
