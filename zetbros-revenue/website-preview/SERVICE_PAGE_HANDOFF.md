# Zetbros /agents service-page draft

Prepared 3 October 2026. Implemented locally and unpublished. Automated checks
pass; browser UI verification is blocked and remains incomplete.

## Source and changes

- Repository: `Logan17de/zetbros.com`
- Verified main baseline: `9c7e4b366d3428f43c6b4fa5dc420599588de237`
- Add `app/agents/page.tsx` and `app/agents/agents.module.css`
- Update `tests/site.test.mjs` to verify the draft and retain existing checks
- Narrow patch: `service-page.patch` (281 additions, 2 deletions, 3 files)
- No website commit, push, pull request or deployment was performed

The page reuses the existing logo, Inter font, shared mineral/teal canvas,
glass treatment, practice-page navigation and ContactTrigger. No dependencies
or contact behavior were changed.

## What the draft says

The proposed service starts with a paid Integration Readiness Audit, which can
recommend a no-go. A qualified Agent Integration Pilot is limited to one
documented API, one existing customer account or tenant, one named operator,
one agreed AI client/version and up to three read-only tools.

The illustrative status-brief workflow is explicitly an example, not a customer
case study. Data-processing destinations need customer approval. Email sends,
external notifications, business-record changes, new OAuth infrastructure,
multi-user authorization and ongoing operation are excluded from the entry
pilot. Production use needs a separate release decision.

The page contains no numeric price, delivery promise, testimonial, synthetic
approval replay or live Mail integration. Fees, schedule and commercial/data
terms remain subject to qualification and agreement.

The visible unpublished label and `noindex, nofollow` metadata are deliberate.
The page is excluded from the public sitemap and has no inbound link from the
existing public routes. Noindex is not an access or privacy control.

## Final verification

- Locked dependency installation: passed with a writable temporary npm cache
- `npm run build`: passed; `/agents` exported as static HTML
- `npm run lint` (TypeScript): passed against the final source
- `npm test`: 6 Worker tests passed, 0 failed
- `node --test tests/site.test.mjs`: 19 static tests passed, 0 failed
- Whitespace checks and reverse patch applicability: passed
- Original logo/icons, package lock, shared styling, contact components,
  Worker source and public sitemap source: unchanged

The static checks cover canonical/share metadata, resolved internal links,
owner-identity exclusion, original assets, the draft's non-indexable status,
bounded service wording and reuse of the existing contact route. Source
inspection finds no new integration transport or server route, and exported
initial HTML contains no form, upload, iframe or embedded proof.

These are recorded automated results for the final implementation. Worker tests
use simulated failure events and do not represent live contact submissions.
The full website checkout, dependency directories, build output and installation
logs are not included in this narrow patch package.

## Browser limitation

Browser UI verification was blocked by a local-preview access error
(`net::ERR_BLOCKED_BY_CLIENT`). No visual or interaction pass is claimed.

Consequently, 320/390/768/1440 responsive rendering, actual contact opening,
subject prefills, focus/Escape/close/return-focus behavior, reduced motion,
Back/Forward and browser console/network behavior were not verified. No
screenshot of the implemented page is available. Source review is not a
substitute for those UI checks. No contact form was submitted.

## Review and next decision

Apply `service-page.patch` to a clean checkout of the recorded baseline after
checking it with `git apply --check`. The patch includes both new source files and the test changes. It requires the
existing website source at the recorded baseline; it is not a standalone site.

Complete browser QA in a supported local preview, review the service wording
and resolve commercial/data terms before deciding on publication. Publication,
sitemap inclusion, inbound navigation and removal of draft/noindex labels are
separate decisions. This package itself does not publish a service on Zetbros.
