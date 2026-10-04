# Eego

A private English notebook for Mayuna, with B1–C2 vocabulary and grammar, Japanese explanations, typed cloze practice, saved history, and spaced review.

Live app: https://eego.leedaway.chatgpt.site/

## Hosting and storage

- ChatGPT Sites serves HTML, CSS, JavaScript and a same-origin API proxy.
- The `eego-api` Worker runs in the owner's Cloudflare account, with an `eego` D1 database.
- The Oracle VM runs the Python question generator as `eego-codex.service`, enabled at boot with automatic restart.
- Generation uses the owner's saved ChatGPT OAuth login through the official Codex CLI. No paid runtime API key is required. Codex account limits and login validity still apply.
- The app has one password account, no registration endpoint, and 30-day Secure / HttpOnly / SameSite cookies. Passwords use a secret pepper and PBKDF2-SHA256. The login password is provisioned privately and is absent from source.

Saved lessons and progress remain available when the generator is temporarily offline. The web app and database continue running when the owner's PC is off. New questions are queued, generated, checked by a second Codex pass, and saved back to D1 over HTTPS.

## Source layout

`web/` contains the responsive interface. `hosting/index.js` is the Sites proxy. `cloudflare/index.js` is the D1 API and `cloudflare/schema.sql` is its initial schema. `data/seed.json` contains 40 original learning items and 80 starter questions. `worker/worker.py` contains the bounded generation service, with no third-party Python dependencies.

## Build and checks

Requirements: Node.js and Python 3.11+.

```sh
npm run build
npm test
```

The build embeds the static assets in `dist/server/index.js`. Tests cover routing, cross-site request rejection, cookie isolation, strict generated-question validation and Codex process isolation. The POSIX executable and file-permission checks run on the VM; those two checks are skipped on Windows.

## Runtime configuration

The Cloudflare Worker needs a `DB` binding and three secret bindings: `AUTH_PEPPER`, `SITE_TOKEN` and `WORKER_TOKEN_HASH`. The last value is the SHA-256 digest of the private 64-hex worker token. The Sites runtime needs `EEGO_PROXY_TOKEN`, equal to `SITE_TOKEN`. Never place these values in the frontend or a Git commit.

`cloudflare/wrangler.jsonc` identifies the deployed Worker and D1 database. Apply the schema only to a new empty database, then import `data/seed.json` and provision the single account with a privately generated salt and password hash. Hashing is HMAC-SHA256(secret pepper, password), then PBKDF2-SHA256 with 100,000 iterations, the hex-decoded salt, and a 32-byte result.

For a fresh runner, install the official Codex CLI and authenticate its private `CODEX_HOME` using `codex login`. A trusted private runner may be securely seeded from an existing CLI login; protect `auth.json` as a password. Keep the runner's writable authentication directory outside the source checkout.

Copy `worker/worker.env.example` to `/home/ubuntu/.config/eego/worker.env`, fill the private worker token and set mode 600. Put `worker.py` at `/home/ubuntu/eego/worker.py`. Install the service example as `/etc/systemd/system/eego-codex.service`, then run:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now eego-codex.service
systemctl status eego-codex.service
journalctl -u eego-codex.service --since today
```

The runner opens no inbound port. It polls the authenticated Worker queue. Its token can claim jobs and save validated questions but cannot sign in, read personal answers, export history or change passwords. Codex receives lesson targets and previous question sentences; it never receives the app password, session cookie, worker token or personal study history.

Generation allows 5, 10 or 20 questions per batch, at most two pending batches and 150 requested questions per rolling 24 hours. Incorrect or uncertain answers enter the weak list. Recovery requires three scheduled successes on different sentences; immediate repeats do not advance recovery. Generated content can still contain errors; reporting a question hides it from future practice.

## Publishing

Sites publishing uses the registered project in `.openai/hosting.json`, its native source workflow, and an archive built from the exact pushed source commit. The custom login page is reachable from the shared URL; every study API request requires the account session. Only the Sites proxy has the private backend proxy credential.

The previous project backend was removed. This version uses Cloudflare D1 and the VM service.
