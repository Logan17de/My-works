PRAGMA foreign_keys = ON;
CREATE TABLE users (id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL, salt TEXT NOT NULL);
CREATE TABLE sessions (token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), expires_at INTEGER NOT NULL);
CREATE TABLE login_limits (key TEXT PRIMARY KEY, attempts INTEGER NOT NULL, window INTEGER NOT NULL);
CREATE TABLE items (id TEXT PRIMARY KEY, kind TEXT NOT NULL, level TEXT NOT NULL, label TEXT NOT NULL, meaning_ja TEXT NOT NULL, notes_ja TEXT NOT NULL, examples TEXT NOT NULL, source TEXT NOT NULL);
CREATE TABLE jobs (id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), kind TEXT NOT NULL, level TEXT NOT NULL, count INTEGER NOT NULL, topic TEXT NOT NULL, targets TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'queued', created_at INTEGER NOT NULL, lease TEXT, lease_expires INTEGER, error TEXT, question_count INTEGER NOT NULL DEFAULT 0);
CREATE TABLE questions (id TEXT PRIMARY KEY, item_id TEXT NOT NULL REFERENCES items(id), sentence TEXT NOT NULL, options TEXT NOT NULL, answer TEXT NOT NULL, hint_ja TEXT NOT NULL, explanation_ja TEXT NOT NULL, translation_ja TEXT NOT NULL, source TEXT NOT NULL, fingerprint TEXT NOT NULL UNIQUE, job_id TEXT REFERENCES jobs(id));
CREATE INDEX questions_item ON questions(item_id);
CREATE TABLE practice (id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), questions TEXT NOT NULL, expires_at INTEGER NOT NULL);
CREATE TABLE mastery (user_id TEXT NOT NULL REFERENCES users(id), item_id TEXT NOT NULL REFERENCES items(id), seen INTEGER NOT NULL DEFAULT 0, correct INTEGER NOT NULL DEFAULT 0, weak INTEGER NOT NULL DEFAULT 0, recovery INTEGER NOT NULL DEFAULT 0, next_due INTEGER NOT NULL, last_question TEXT, PRIMARY KEY(user_id,item_id));
CREATE TABLE attempts (id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), session_id TEXT NOT NULL REFERENCES practice(id), question_id TEXT NOT NULL REFERENCES questions(id), item_id TEXT NOT NULL REFERENCES items(id), choice TEXT NOT NULL, correct INTEGER NOT NULL, unsure INTEGER NOT NULL, created_at INTEGER NOT NULL, feedback TEXT NOT NULL, UNIQUE(session_id,question_id));
CREATE INDEX attempts_user ON attempts(user_id,created_at);
CREATE TABLE reports (user_id TEXT NOT NULL REFERENCES users(id), question_id TEXT NOT NULL REFERENCES questions(id), PRIMARY KEY(user_id,question_id));
CREATE TABLE worker_status (id INTEGER PRIMARY KEY CHECK(id=1), ready INTEGER NOT NULL DEFAULT 0, last_seen INTEGER NOT NULL DEFAULT 0);
INSERT INTO worker_status(id) VALUES(1);
-- On existing installations, apply migrations/0002_live_practice.sql instead.
CREATE TABLE live_practice (session_id TEXT PRIMARY KEY REFERENCES practice(id), job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id), total_count INTEGER NOT NULL CHECK(total_count BETWEEN 1 AND 20), seen_position INTEGER NOT NULL DEFAULT -1);
CREATE TABLE codex_tasks (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES practice(id), type TEXT NOT NULL CHECK(type IN ('question','answer')), position INTEGER NOT NULL, question_id TEXT REFERENCES questions(id), choice TEXT NOT NULL DEFAULT '', unsure INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'queued', created_at INTEGER NOT NULL, lease TEXT, lease_expires INTEGER, result TEXT, error TEXT, UNIQUE(session_id,type,position));
CREATE INDEX codex_tasks_queue ON codex_tasks(type,state,created_at);
