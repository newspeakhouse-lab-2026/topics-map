# faculty-topic-map

A vibe-coded graph visualiser of **Fellowship Candidates × Topics** for the Newspeak
House 2026-27 forum, built only from public data, to explore topics we engaged
with that could become sessions.

Output: **`map.html`** — a single self-contained file (d3 and the dataset are
inlined). Open it from disk, or drop it on any static host. No server, no
network, no build step at view time.

```
python3    scripts/fetch.py    # topic.forum -> data/raw.json      (network)
python3.11 scripts/embed.py    # bodies -> data/embeddings.json     (model)
python3    scripts/build.py    # raw + embeddings -> map.html       (offline)
```

`fetch.py` also writes two **gitignored** files: `data/bodies.json` (topic
bodies normalised to plain text, for the semantic layer) and
`data/cache/names.json` (the name list `build.py` checks against). Body text
routinely names members, so it must never reach a tracked file — see Privacy.

`embed.py` is the **only** stage with third-party dependencies, so `build.py`
stays stdlib-only and offline. It needs Python 3.11+ (`torch`/`onnxruntime`
publish no cp314 wheels) and uses `fastembed` (ONNX Runtime, ~10x smaller than
torch) with `sentence-transformers/all-MiniLM-L6-v2`:

```
python3.11 -m pip install fastembed scikit-learn
python3.11 scripts/embed.py
```

It emits `data/embeddings.json` (gitignored): 2D coords, kNN neighbours,
k-means clusters with c-TF-IDF terms, and int8 vectors. `build.py` inlines only
the compact parts (~7 KB).

Hover a node to isolate its neighbourhood, click to pin, double-click a topic
to open it on topic.forum. The sidebar ranks topics at or above the threshold
and, for whatever is focused, lists related topics by shared supporters.

The **search box** ranks all 150 topics with BM25 over title + body, dimming
everything that does not match. Matches reveal their titles on the graph even
when they sit below the usual label cutoff, so results are readable without a
sideways glance at the sidebar. It is lexical, not semantic: `build.py` inlines
the whole index, so the page still works offline, and only the query is analysed
in the browser. Hits below the heart threshold still appear in the list and open
on topic.forum when they are not on the graph.

Switch **layout** from `force` to `semantic` to pin topics at their t-SNE
coordinates from the title+body embedding, coloured by k-means cluster (the key
shows each cluster's c-TF-IDF terms; hover one to isolate it and reveal that
cluster's titles). The Candidate/Topic key items isolate that node type on hover
too, but deliberately leave labelling alone. Candidates are
drawn at a **uniform radius** and dropped at the centre of the topics they
hearted, so the candidate→topic structure survives the switch. Focusing a topic
also lists its **nearest neighbours in meaning** — cosine neighbours of the
embedding — which, unlike the supporter panel, still says something for a topic
only one person hearted. View choices (layout, metric, threshold, labels) are
tucked into `localStorage` and restored on reload; `reset view` clears them.

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

The name-bearing intermediates are all gitignored: `data/cache/` (raw API
responses), `data/bodies.json` (normalised bodies — these name members even
more often than hearts do) and `data/embeddings.json`. `embed.py` also strips
name tokens from the c-TF-IDF cluster terms, so a cluster keyword can never be
a member's name.

As a backstop, `build.py` refuses to write `map.html` if any name, person slug
or name token from `data/cache/names.json` appears anywhere readable in the
payload. That guard is what makes "body text never reaches the artefact" a
checked invariant rather than an intention. Fields that legitimately hold
opaque blobs (future embedding vectors) must be listed in `OPAQUE_KEYS` so the
scan skips them — never for readable text.

## Notes that shaped the design

- **The semantic layer is precomputed, never live.** `embed.py` does the model
  work once; `build.py` inlines coordinates and neighbour lists, so the page
  stays offline and `build.py` stays dependency-free. Embeddings are fully
  deterministic here (fixed seed, PCA init, ONNX): re-running produced
  byte-identical vectors, coords, clusters and neighbours.
- **The semantic layout is relaxed, not raw t-SNE.** Topics are pinned, so the
  simulation's collide force cannot separate them, and t-SNE routinely drops
  topics on top of each other; scaling the projection up just clips nodes.
  Instead `relaxLayout()` pushes overlapping pairs apart using worst-case radii
  — so the guarantee holds for *every* size metric — then refits the cloud to
  the viewport. The result is independent of metric and threshold, at the cost
  of a small distortion of a projection that was never faithful to begin with.
- **Be honest about the clusters.** k-means on MiniLM embeddings separates the
  150 topics into 7 groups with a **silhouette of 0.04** — far apart in meaning
  is not the same as well-separated in vector space. Treat the cluster labels as
  navigation, and the kNN neighbour lists as the trustworthy signal.
- **The two "related" panels answer different questions** and both are kept:
  overlap of *supporters* (Jaccard) measures shared curation, while semantic
  neighbours measure shared *meaning* and need no shared supporters at all —
  which matters when the median topic has only 2 hearts.

- **Search is a precomputed BM25 index, not a query encoder.** Inlining the
  index keeps the single file offline; the cost is that the query analyser
  exists twice -- `stem`/`analyze` in `build.py` and their mirror in
  `web/template.html` -- and the two must stay in step.
- **Name tokens are dropped from the index.** Bodies name members, so
  `build.py` removes any index term that is a known name/slug before inlining
  it -- otherwise the privacy guard would (correctly) refuse to build.

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

## Deploying

`map.html` is the whole site. The included GitHub Actions workflow
(`.github/workflows/pages.yml`) publishes it to GitHub Pages as `index.html` on
every push to `main`.

## Licence

Code is MIT (see `LICENSE`). `map.html` inlines d3 v7.9.0, which is
BSD-3-Clause — see `vendor/d3.LICENSE`.

## Caveats

- n=14 is small. Treat any ranking as a conversation aid, not a verdict.
- Anonymous access only; no tokens are used or stored.
- The committed `map.html` is the artefact: rebuilding it needs the gitignored
  intermediates, so a fresh clone produces a titles-only map with no semantic
  layer until `fetch.py` and `embed.py` have run.
