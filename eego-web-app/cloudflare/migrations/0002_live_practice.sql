-- Additive migration: keep all existing lessons, accounts and study records.
CREATE TABLE IF NOT EXISTS live_practice (
 session_id TEXT PRIMARY KEY REFERENCES practice(id),
 job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id),
 total_count INTEGER NOT NULL CHECK(total_count BETWEEN 1 AND 20),
 seen_position INTEGER NOT NULL DEFAULT -1
);
CREATE TABLE IF NOT EXISTS codex_tasks (
 id TEXT PRIMARY KEY,
 session_id TEXT NOT NULL REFERENCES practice(id),
 type TEXT NOT NULL CHECK(type IN ('question','answer')),
 position INTEGER NOT NULL,
 question_id TEXT REFERENCES questions(id),
 choice TEXT NOT NULL DEFAULT '',
 unsure INTEGER NOT NULL DEFAULT 0,
 state TEXT NOT NULL DEFAULT 'queued',
 created_at INTEGER NOT NULL,
 lease TEXT,
 lease_expires INTEGER,
 result TEXT,
 error TEXT,
 UNIQUE(session_id,type,position)
);
CREATE INDEX IF NOT EXISTS codex_tasks_queue ON codex_tasks(type,state,created_at);
