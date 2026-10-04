# Eego continuation

Canonical user repo: `Logan17de/My-works/eego-web-app`.
Sites project identity: `.openai/hosting.json`.
Supabase project: `jxvabaqswqembehxligi`; private schema: `eego`; function: `eego-api`.

The complete source has been restored. The authenticated Edge backend has been published and 40 starter items / 80 questions imported additively. The prior backend/import safety blocks no longer occurred on the authorized deployment calls in this task. Preserve the existing Mayuna account and its password.

The remaining external dependency is the owner's real Codex machine. Harness failed when checked. Do not claim Codex is connected based on mock tests or a queued job. No API-key fallback exists.

On the authenticated Codex machine, run `python3 worker/connect.py --start` from this folder. This provisions a scoped token using the existing Eego login and stores it outside the repo. Do not print passwords, cookies, auth.json, or worker tokens. The worker claims only its own user's jobs. Never remove isolation flags to make an incompatible CLI start.

Verify a five-question request reaches completed, the saved questions appear in practice, and the offline indicator returns after the worker stops. Record non-secret evidence. Keep source, hosted Site and this status document aligned when work resumes.

Existing users' study history must never be used as test fixtures. The database test rolls back its fixtures. Any live test account and all related jobs/workers/history must be removed after testing.

The older Supabase function `eego` and public `eego_*` legacy tables were left intact because other prior work may still reference them. This app uses only `eego-api` and the private schema. Do not delete legacy content without checking usage and authorization.
