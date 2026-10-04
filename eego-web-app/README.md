# Eego

A private English notebook for Mayuna, with B1–C2 vocabulary and grammar, Japanese explanations, typed cloze practice, saved history, and spaced review.

Live app: https://eego.leedaway.chatgpt.site/

## Hosting and storage

- ChatGPT Sites serves HTML, CSS, JavaScript and a same-origin API proxy.
- The `eego-api` Worker runs in the owner's Cloudflare account, with an `eego` D1 database.
- The Oracle VM runs the Python question generator as `eego-codex.service`, enabled at boot with automatic restart.
- Generation uses the owner's saved ChatGPT OAuth login through the official Codex CLI. No paid runtime API key is required. Codex account limits and login validity still apply.
- The app has one password account, no registration endpoint, and 30-day Secure / HttpOnly / SameSite cookies. Passwords use a secret pepper and PBKDF2-SHA256. The login password is provisioned privately and is absent from source.

Saved lessons and progress remain readable when the generator is temporarily offline. The web app, database and VM generator continue running when the owner's PC is off. Both generation and answer checking use the owner's Codex OAuth login with GPT-6 Luna and low reasoning effort.

New practice is incremental. Starting a 5-, 10- or 20-question session queues only question 1. Each generation call produces one cloze sentence and a short Japanese clue, with no answer, explanation or translation. Once the browser displays a question, it requests exactly one next question. Reading question 2 can queue question 3 only after question 1's answer has been checked.

Submit creates a separate Codex task containing the current question, lesson and learner answer. Codex judges the answer in context, accepts valid alternatives, and returns the correct completion plus explanations, specific learning suggestions and a new example in both English and Japanese. It also translates the original completed sentence into Japanese. Feedback explains why this answer works and why an incorrect choice does not; suggestions reinforce a transferable word pairing or grammar pattern. All feedback is generated in that single check call, after Submit. Feedback and progress are committed together, once per submitted question. Next appears only after feedback arrives. Answer checks have their own worker lane so they can run while the next question is being prepared. Basic local format checks replace the old second AI review. Saved starter questions also use Codex when an answer is submitted.

## Source layout

`web/` contains the responsive interface. `hosting/index.js` is the Sites proxy. `cloudflare/index.js` is the D1 API and `cloudflare/schema.sql` is its initial schema. `data/seed.json` contains 40 original learning items and 80 starter questions. `worker/worker.py` contains the bounded generation service, with no third-party Python dependencies.

## Build and checks

Requirements: Node.js and Python 3.11+.

```sh
npm run build
npm test
```

The build embeds the static assets in `dist/server/index.js`. Tests cover routing, cookie isolation, generation only after reading, checking only after Submit, concurrent queues, duplicate submissions/completions, Japanese feedback, retries, session ownership and Codex process isolation. API checks execute the real handler against SQLite with D1's prepared-statement interface. Node 22.13+ is required for these checks. The POSIX executable and file-permission checks run on the VM; those two checks are skipped on Windows.

## Runtime configuration

The Cloudflare Worker needs a `DB` binding and three secret bindings: `AUTH_PEPPER`, `SITE_TOKEN` and `WORKER_TOKEN_HASH`. The last value is the SHA-256 digest of the private 64-hex worker token. The Sites runtime needs `EEGO_PROXY_TOKEN`, equal to `SITE_TOKEN`. Never place these values in the frontend or a Git commit.

`cloudflare/wrangler.jsonc` identifies the deployed Worker and D1 database. Apply the schema only to a new empty database, then import `data/seed.json` and provision the single account with a privately generated salt and password hash. Existing 2.0 installations apply only `cloudflare/migrations/0002_live_practice.sql`; the migration preserves all existing records. Hashing is HMAC-SHA256(secret pepper, password), then PBKDF2-SHA256 with 100,000 iterations, the hex-decoded salt, and a 32-byte result.

For a fresh runner, install the official Codex CLI and authenticate its private `CODEX_HOME` using `codex login`. A trusted private runner may be securely seeded from an existing CLI login; protect `auth.json` as a password. Keep the runner's writable authentication directory outside the source checkout.

Copy `worker/worker.env.example` to `/home/ubuntu/.config/eego/worker.env`, fill the private worker token and set mode 600. Put `worker.py` at `/home/ubuntu/eego/worker.py`. Install the service example as `/etc/systemd/system/eego-codex.service`, then run:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now eego-codex.service
systemctl status eego-codex.service
journalctl -u eego-codex.service --since today
```

The runner opens no inbound port and polls separate question and answer queues every 3 seconds. Its token can claim narrowly scoped tasks and save results but cannot sign in, export history or change passwords. Codex receives the current lesson, question and submitted answer for checks; it never receives the app password, session cookie, worker token or full study history. The practice screen refreshes pending work every 2 seconds and preserves typed answers while the next question arrives. Queue and model timing logs contain no prompts or credentials.

Generation allows sessions of 5, 10 or 20 questions, at most two active sessions and 150 requested questions per rolling 24 hours. Failed tasks can be retried; expired leases cannot overwrite a retry. Save & leave preserves a session and any pending answer check. Incorrect or uncertain answers enter the weak list. Recovery requires three scheduled successes on different sentences; immediate repeats do not advance recovery. Reporting a question hides it from future practice.

## Publishing

Sites publishing uses the registered project in `.openai/hosting.json`, its native source workflow, and an archive built from the exact pushed source commit. The custom login page is reachable from the shared URL; every study API request requires the account session. Only the Sites proxy has the private backend proxy credential.

The previous project backend was removed. This version uses Cloudflare D1 and the VM service.
