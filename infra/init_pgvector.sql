-- =============================================================================
-- OmniSignal / FiberOrbit — PGVector Schema
-- Applies automatically on first container start via docker-entrypoint-initdb.d
-- =============================================================================

-- Enable pgvector extension (must be done before creating vector columns)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- for fast LIKE / similarity searches


-- =============================================================================
-- Roles & Permissions (module-level RBAC)
-- =============================================================================

-- HR data is siloed: only the hr_agent role can read hr_docs
CREATE ROLE hr_agent        LOGIN PASSWORD 'hr_agent_pw'        NOSUPERUSER;
CREATE ROLE marketing_agent LOGIN PASSWORD 'marketing_agent_pw' NOSUPERUSER;
CREATE ROLE rag_admin       LOGIN PASSWORD 'rag_admin_pw'       NOSUPERUSER;

GRANT CONNECT ON DATABASE fiberorbit_db TO hr_agent, marketing_agent, rag_admin;


-- =============================================================================
-- HR Module — hr_docs
-- ONLY the hr_agent role can SELECT. No marketing agent access.
-- =============================================================================

CREATE TABLE IF NOT EXISTS hr_docs (
    id          BIGSERIAL    PRIMARY KEY,
    content     TEXT         NOT NULL,
    embedding   vector(1536),
    metadata    JSONB        NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- IVFFlat index for ANN search (lists = 100 is good for < 1M vectors)
CREATE INDEX IF NOT EXISTS idx_hr_docs_embedding
    ON hr_docs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- JSONB index for metadata pre-filtering (the 18% precision improvement)
CREATE INDEX IF NOT EXISTS idx_hr_docs_metadata
    ON hr_docs USING gin (metadata jsonb_path_ops);

-- Full-text search index for keyword retrieval
CREATE INDEX IF NOT EXISTS idx_hr_docs_fts
    ON hr_docs USING gin (to_tsvector('english', content));

-- RBAC: hr_agent can SELECT; marketing_agent cannot access this table at all
GRANT SELECT, INSERT, UPDATE, DELETE ON hr_docs TO hr_agent, rag_admin;
GRANT USAGE, SELECT ON SEQUENCE hr_docs_id_seq TO hr_agent, rag_admin;


-- =============================================================================
-- Marketing Modules
-- =============================================================================

-- Campaign analytics documents
CREATE TABLE IF NOT EXISTS campaign_docs (
    id          BIGSERIAL    PRIMARY KEY,
    content     TEXT         NOT NULL,
    embedding   vector(1536),
    metadata    JSONB        NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_campaign_docs_embedding
    ON campaign_docs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_campaign_docs_metadata
    ON campaign_docs USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_campaign_docs_fts
    ON campaign_docs USING gin (to_tsvector('english', content));

GRANT SELECT, INSERT, UPDATE, DELETE ON campaign_docs TO marketing_agent, rag_admin;
GRANT USAGE, SELECT ON SEQUENCE campaign_docs_id_seq TO marketing_agent, rag_admin;


-- Fiber network coverage and build plans
CREATE TABLE IF NOT EXISTS fiber_network_docs (
    id          BIGSERIAL    PRIMARY KEY,
    content     TEXT         NOT NULL,
    embedding   vector(1536),
    metadata    JSONB        NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fiber_network_docs_embedding
    ON fiber_network_docs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_fiber_network_docs_metadata
    ON fiber_network_docs USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_fiber_network_docs_fts
    ON fiber_network_docs USING gin (to_tsvector('english', content));

GRANT SELECT, INSERT, UPDATE, DELETE ON fiber_network_docs TO marketing_agent, rag_admin;
GRANT USAGE, SELECT ON SEQUENCE fiber_network_docs_id_seq TO marketing_agent, rag_admin;


-- Competitor intelligence
CREATE TABLE IF NOT EXISTS competitive_docs (
    id          BIGSERIAL    PRIMARY KEY,
    content     TEXT         NOT NULL,
    embedding   vector(1536),
    metadata    JSONB        NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_competitive_docs_embedding
    ON competitive_docs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_competitive_docs_metadata
    ON competitive_docs USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_competitive_docs_fts
    ON competitive_docs USING gin (to_tsvector('english', content));

GRANT SELECT, INSERT, UPDATE, DELETE ON competitive_docs TO marketing_agent, rag_admin;
GRANT USAGE, SELECT ON SEQUENCE competitive_docs_id_seq TO marketing_agent, rag_admin;


-- SEO / content strategy documents
CREATE TABLE IF NOT EXISTS seo_docs (
    id          BIGSERIAL    PRIMARY KEY,
    content     TEXT         NOT NULL,
    embedding   vector(1536),
    metadata    JSONB        NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_seo_docs_embedding
    ON seo_docs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_seo_docs_metadata
    ON seo_docs USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_seo_docs_fts
    ON seo_docs USING gin (to_tsvector('english', content));

GRANT SELECT, INSERT, UPDATE, DELETE ON seo_docs TO marketing_agent, rag_admin;
GRANT USAGE, SELECT ON SEQUENCE seo_docs_id_seq TO marketing_agent, rag_admin;


-- Financial models, Calix data, copper/fiber build financials
CREATE TABLE IF NOT EXISTS financial_docs (
    id          BIGSERIAL    PRIMARY KEY,
    content     TEXT         NOT NULL,
    embedding   vector(1536),
    metadata    JSONB        NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_financial_docs_embedding
    ON financial_docs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_financial_docs_metadata
    ON financial_docs USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_financial_docs_fts
    ON financial_docs USING gin (to_tsvector('english', content));

GRANT SELECT, INSERT, UPDATE, DELETE ON financial_docs TO marketing_agent, rag_admin;
GRANT USAGE, SELECT ON SEQUENCE financial_docs_id_seq TO marketing_agent, rag_admin;


-- =============================================================================
-- Fidelity — golden baseline table
-- Stores validated Q&A pairs used by the report_validator fidelity gate.
-- =============================================================================

CREATE TABLE IF NOT EXISTS fidelity_baselines (
    id              BIGSERIAL   PRIMARY KEY,
    query_hash      TEXT        NOT NULL UNIQUE,   -- SHA256 of canonical query
    query           TEXT        NOT NULL,
    expected_answer TEXT        NOT NULL,
    agent           TEXT        NOT NULL,
    numeric_values  JSONB       NOT NULL DEFAULT '{}',   -- key figures to check
    created_by      TEXT        NOT NULL,
    validated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    active          BOOLEAN     NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_fidelity_agent ON fidelity_baselines (agent, active);

GRANT SELECT ON fidelity_baselines TO marketing_agent, hr_agent;
GRANT SELECT, INSERT, UPDATE, DELETE ON fidelity_baselines TO rag_admin;
GRANT USAGE, SELECT ON SEQUENCE fidelity_baselines_id_seq TO rag_admin;


-- =============================================================================
-- Trigger: auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'hr_docs', 'campaign_docs', 'fiber_network_docs',
        'competitive_docs', 'seo_docs', 'financial_docs'
    ]
    LOOP
        EXECUTE format(
            'CREATE TRIGGER trg_%s_updated_at
             BEFORE UPDATE ON %s
             FOR EACH ROW EXECUTE FUNCTION update_updated_at()',
            tbl, tbl
        );
    END LOOP;
END;
$$;
