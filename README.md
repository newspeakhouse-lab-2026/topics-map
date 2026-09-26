# faculty-topic-map

A vibe-coded graph visualiser of **Fellowship Candidates × Topics** for the Newspeak
House 2026-27 forum, built only from public data, to explore topics we engaged
with that could become sessions.

Output: **`map.html`** — a single self-contained file (d3 and the dataset are
inlined). Open it from disk, or drop it on any static host. No server, no
network, no build step at view time.

```
python3 scripts/fetch.py     # topic.forum -> data/raw.json   (network)
python3 scripts/build.py     # data/raw.json -> map.html      (offline)
```

Hover a node to isolate its neighbourhood, click to pin, double-click a topic
to open it on topic.forum. The sidebar ranks topics at or above the threshold
and, for whatever is focused, lists related topics by shared supporters.

## The four metrics

The forum returns `weightedScore`, `l2Score` and `devotionScore` as `null` to
anonymous readers, but they are deterministic functions of the edge list, which
*is* public via `topicFeed(heartedBy:)`. `build.py` reproduces them locally,
ported from the app's own `packages/shared/src/hearts.ts`:

```
weight(candidate) = 1 / (published topics they hearted)

raw       = Σ 1            every heart equal           (== heartCount)
l1        = Σ weight       one unit of attention each  (== weightedScore)
l2        = Σ 1/√total     attention discounted by √   (== l2Score)
devotion  = l1 / raw       mean supporter devotion     (== devotionScore)
```

Each candidate spreads a total influence of 1 across their hearts, so hearting
fewer topics makes each heart count for more. `build.py` refuses to build
unless its `raw` reproduces the server's `heartCount` on every topic, so the
derived scores can be trusted.

Default view is `l1` with a floor of 2 hearts. Note `devotion` is an *intensity*
measure, not support — one heart from someone who hearted nothing else scores a
perfect 1.0, so it makes a poor session threshold.

## Privacy

No full name, person slug or raw user id appears in any tracked file:

- Candidates are shown as initials (`MK`, `JNC`, `N`) and keyed by a short
  opaque token (`sha256(userId)[:10]`). Names are used only to derive initials,
  then discarded.
- Topic links use `/f/<forum>/t/<topic-slug>` — the forum's middle path segment
  is decorative and 308-redirects to the canonical URL, so no proposer slug is
  needed.
- `build.py` fails the build if a name, slug or non-token id reaches the
  artefact.

**Be clear-eyed about the limit.** Tokens defeat casual reading — `grep` gets you
nowhere — but `forumPeople` is a public query returning userId + name, so anyone
can hash every member's id and match ours. The heart data is public and
attributable anyway (`topicFeed(heartedBy:)` works anonymously). So initials are
a **presentation choice, not an anonymity guarantee**, and within a 14-person
cohort an initial plus a heart pattern may still identify someone.

`data/cache/` holds raw API responses (which do contain names) and is gitignored
— do not commit it.

## Notes that shaped the design

- **No `seen`/view data is public.** The app tracks it internally but exposes it
  only as per-viewer read state, or via the host/admin-only
  `dashboard { electorActivity }`, which returns `null` to us. A likes/seen ratio
  isn't computable without host credentials — the four normalisations above are
  the app's own answer to the same problem.
- **Global clustering isn't meaningful at n=14.** The most active candidates
  heart near-identical sets (Jaccard 0.30–0.53), so the co-heart graph is one
  blob at any sensible threshold. The tool offers *local* overlap instead.
- **Host hearts exist but are invisible and inert** — stored separately, not
  counted in `heartCount`, admin-only. `heartCount` is elector hearts only.
- **The forum resets hearts termly** via a `heartsCountFrom` cutoff, so this is a
  one-term snapshot by design — re-run both scripts each term.
- **The API silently caps `limit` at 50** for both `topicFeed` and `heartedBy`,
  so lists are offset-paginated and queries are batched with GraphQL aliases
  (~4 requests per refresh instead of ~30). Denominators count published topics
  only, matching the app's rule.

## Caveats

- n=14 is small. Treat any ranking as a conversation aid, not a verdict.
- Anonymous access only; no tokens are used or stored.
- `map.html` inlines d3 v7.9.0 (BSD-3-Clause) from `vendor/`.
