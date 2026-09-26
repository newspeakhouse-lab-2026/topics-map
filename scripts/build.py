#!/usr/bin/env python3
"""Build a self-contained map.html from data/raw.json.

Recomputes the forum's four heart normalisations locally -- the API returns
`weightedScore`, `l2Score` and `devotionScore` as null for anonymous readers,
but they are deterministic functions of the edge list, which *is* public.

The maths is a port of `packages/shared/src/hearts.ts` from the app itself:

    weight(elector)      = 1 / (published topics that elector hearted)
    raw                  = sum(1)                     for each heart   (L-infinity)
    l1 == weightedScore  = sum(weight)                for each heart
    l2 == l2Score        = sum(1 / sqrt(total))       for each heart
    devotion             = l1 / raw                   (mean supporter devotion)

d3 and the dataset are inlined, so the result is a single portable file that
works over file:// with no network and no build step at view time.

Usage:
    python3 scripts/build.py

Reads data/raw.json + web/template.html + vendor/d3.v7.min.js.
Writes map.html.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys

# Must match scripts/fetch.py::TOKEN_LEN.
TOKEN_LEN = 10

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "data", "raw.json")
TEMPLATE_PATH = os.path.join(ROOT, "web", "template.html")
D3_PATH = os.path.join(ROOT, "vendor", "d3.v7.min.js")
OUT_PATH = os.path.join(ROOT, "map.html")
# Optional name list written by fetch.py (gitignored). Present on a normal dev
# checkout, absent on a fresh clone -- the guard degrades to a warning then.
NAMES_PATH = os.path.join(ROOT, "data", "cache", "names.json")
# Normalised topic bodies (gitignored, name-bearing). Optional: the search
# index falls back to titles when it is absent, e.g. on a fresh clone.
BODIES_PATH = os.path.join(ROOT, "data", "bodies.json")

# Fields whose string values are opaque blobs (base64/quantised vectors). They
# carry no readable text, so the name guard skips them. Empty until the
# embedding stage lands.
OPAQUE_KEYS: set[str] = set()


def strings_in(value, key=None):
    """Yield every human-readable string in a nested payload."""
    if isinstance(value, str):
        if key not in OPAQUE_KEYS:
            yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from strings_in(v, k)
    elif isinstance(value, list):
        for v in value:
            yield from strings_in(v)


def load_names():
    """The name list fetch.py writes (gitignored), or None if unavailable."""
    try:
        with open(NAMES_PATH) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def name_guard(payload, names) -> list[str]:
    """Names, slugs or name tokens from fetch.py that appear in the payload.

    This is the enforcement behind the privacy promise: it is not enough for
    `build.py` to *intend* not to inline names, it has to fail loudly if one
    ever arrives -- via a body, a top-term, or a field someone adds later.
    Returns [] (with a warning) when the name list is unavailable.
    """
    if not names:
        print("  note: data/cache/names.json absent -- personal-name guard skipped "
              "(run scripts/fetch.py to enable it)")
        return []

    hay = "\n".join(strings_in(payload)).lower()
    wanted = names.get("names", []) + names.get("slugs", []) + names.get("tokens", [])
    hits = []
    for value in dict.fromkeys(v.strip() for v in wanted if v and v.strip()):
        pattern = r"(?<![a-z0-9])" + re.escape(value.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, hay):
            hits.append(value)
    return hits


# --- lexical search ------------------------------------------------------
# BM25 over title + body. The whole index is precomputed here and inlined, so
# the page stays offline; only the *query* is analysed at runtime, by the
# mirror of `stem`/`analyze` in web/template.html. Keep the two in sync.
STOPWORDS = frozenset(
    "a about above after again against all am an and any are as at be because been "
    "before being below between both but by can cannot could did do does doing down "
    "during each few for from further had has have having he her here hers herself "
    "him himself his how i if in into is it its itself just me more most my myself "
    "no nor not now of off on once only or other our ours ourselves out over own "
    "same she should so some such than that the their theirs them themselves then "
    "there these they this those through to too under until up very was we were "
    "what when where which while who whom why will with you your yours yourself "
    "yourselves".split()
)

# Light inflector: plurals and sibilants only. Deliberately not a full Porter
# stemmer -- the goal is that "topic" and "topics" meet, not linguistic
# correctness, and short rules are easy to mirror exactly in JS.
_SUFFIXES = (
    ("ies", "y", 3),
    ("sses", "ss", 2),
    ("shes", "sh", 2),
    ("ches", "ch", 2),
    ("xes", "x", 2),
    ("zes", "z", 2),
)


def stem(word: str) -> str:
    for suffix, replacement, min_base in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_base:
            return word[: len(word) - len(suffix)] + replacement
    if (word.endswith("s") and len(word) > 3
            and not word.endswith(("ss", "us", "is"))):
        return word[:-1]
    return word


def analyze(text: str) -> list[str]:
    return [
        stem(t)
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) >= 2 and t not in STOPWORDS
    ]


def drop_terms(names) -> frozenset:
    """Index terms that must never be emitted: they are member names."""
    if not names:
        return frozenset()
    values = names.get("tokens", []) + names.get("slugs", [])
    return frozenset(v.lower().strip() for v in values if v and v.strip())


def build_search_index(topics, bodies, forbidden) -> dict:
    """Inlined BM25 index over title + body.

    Titles are counted twice: they are short and sharply on-topic, so a title
    hit should outweigh a stray mention deep in a long body. Postings are
    stored flat as [doc, tf, doc, tf, ...] to keep the JSON compact.
    """
    docs: list[int] = []
    index: dict[str, list[int]] = {}
    for i, topic in enumerate(topics):
        text = topic["title"] + " " + topic["title"] + " " + bodies.get(topic["id"], "")
        tf: dict[str, int] = {}
        for term in analyze(text):
            if term in forbidden:
                continue
            tf[term] = tf.get(term, 0) + 1
        docs.append(sum(tf.values()))
        for term, n in tf.items():
            index.setdefault(term, []).extend((i, n))
    total = sum(docs)
    return {
        "k1": 1.2,
        "b": 0.75,
        "avg": round(total / len(docs), 3) if docs else 1.0,
        "docs": docs,
        "index": index,
    }


def main() -> int:
    with open(RAW_PATH) as fh:
        raw = json.load(fh)

    names = load_names()

    topics_by_id = {t["id"]: t for t in raw["topics"]}

    # Candidates who never hearted anything carry no edges; drop them entirely
    # so their initials are not embedded either.
    active = [e for e in raw["electors"] if e["heartedTopicIds"]]
    dropped = len(raw["electors"]) - len(active)

    # Denominator: published topics hearted. fetch.py has already dropped
    # hearts on unpublished topics (and counts a stale heart as no heart at
    # all), so this is just the list length. Keyed by an opaque token -- the
    # dataset carries neither a name nor a raw user id.
    totals = {e["id"]: len(e["heartedTopicIds"]) for e in active}
    weights = {u: (1.0 / n if n else 0.0) for u, n in totals.items()}
    hearted_by = {e["id"]: set(e["heartedTopicIds"]) for e in active}

    topics = []
    for topic in raw["topics"]:
        hearters = [u for u in totals if topic["id"] in hearted_by[u]]
        raw_n = len(hearters)
        l1 = sum(weights[u] for u in hearters)
        l2 = sum(1.0 / math.sqrt(totals[u]) for u in hearters if totals[u])
        topics.append(
            {
                "id": topic["id"],
                "title": topic["title"],
                "slug": topic["slug"],
                "publishedAt": topic.get("publishedAt"),
                "heartCount": topic.get("heartCount") or 0,
                "raw": raw_n,
                "l1": round(l1, 4),
                "l2": round(l2, 4),
                "devotion": round(l1 / raw_n, 4) if raw_n else 0.0,
                "hearters": hearters,
            }
        )

    # --- verification: our `raw` must reproduce the server's `heartCount`
    mismatches = [t for t in topics if t["raw"] != t["heartCount"]]
    if mismatches:
        for t in mismatches[:10]:
            print(f"  MISMATCH {t['title'][:40]!r} ours={t['raw']} server={t['heartCount']}")
        print(
            f"ERROR: {len(mismatches)}/{len(topics)} topics disagree with heartCount; "
            "refusing to build (the edge list is incomplete).",
            file=sys.stderr,
        )
        return 1

    electors = [
        {
            "id": e["id"],
            "initials": e["initials"],
            "total": totals[e["id"]],
            "weight": round(weights[e["id"]], 5),
        }
        for e in active
    ]
    electors.sort(key=lambda e: (-e["total"], e["initials"]))

    payload = {
        "generatedAt": raw["generatedAt"],
        "forum": raw["forum"],
        "electors": electors,
        "topics": topics,
    }

    # Lexical search index. Body text is optional: without data/bodies.json
    # (fresh clone) the index falls back to titles, which still searches -- it
    # just cannot match a phrase that only appears in a body.
    try:
        with open(BODIES_PATH) as fh:
            bodies = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        bodies = {}
        print("  note: data/bodies.json absent -- search index uses titles only")
    forbidden = drop_terms(names)
    payload["search"] = build_search_index(topics, bodies, forbidden)

    with open(TEMPLATE_PATH) as fh:
        html = fh.read()

    # Guard against regressions: nothing identifying may reach the artefact.
    # Every id must be an opaque fixed-length digest, so a raw Clerk user id or
    # a person slug sneaking back in fails the build rather than shipping.
    token_re = re.compile(r"^[0-9a-f]{%d}$" % TOKEN_LEN)
    allowed_elector_keys = {"id", "initials", "total", "weight"}
    allowed_topic_keys = {
        "id", "title", "slug", "publishedAt", "heartCount",
        "raw", "l1", "l2", "devotion", "hearters",
    }
    for e in electors:
        if set(e) != allowed_elector_keys:
            print(f"ERROR: unexpected elector keys {sorted(e)}", file=sys.stderr)
            return 1
        if not token_re.match(e["id"]):
            print(f"ERROR: elector id {e['id']!r} is not an opaque token", file=sys.stderr)
            return 1
        if not (1 <= len(e["initials"]) <= 4 and e["initials"][0].isalpha()):
            print(f"ERROR: suspicious initials {e['initials']!r}", file=sys.stderr)
            return 1
    for t in topics:
        if set(t) != allowed_topic_keys:
            print(f"ERROR: unexpected topic keys {sorted(t)}", file=sys.stderr)
            return 1
        for u in t["hearters"]:
            if not token_re.match(u):
                print(f"ERROR: hearter {u!r} is not an opaque token", file=sys.stderr)
                return 1

    # `</script>` anywhere in inlined JS/JSON would close the tag early.
    d3 = open(D3_PATH).read().replace("</script", "<\\/script")
    data = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")

    # Privacy backstop: no name, person slug or name token may reach map.html.
    leaks = name_guard(payload, names)
    if leaks:
        for value in leaks[:10]:
            print(f"  LEAK {value!r}", file=sys.stderr)
        print(
            f"ERROR: {len(leaks)} personal name(s) found in the payload; "
            "refusing to build (add the field to OPAQUE_KEYS only if it is an "
            "opaque blob, never to hide readable text).",
            file=sys.stderr,
        )
        return 1

    html = html.replace("/*__D3__*/", d3).replace("__DATA__", data)

    with open(OUT_PATH, "w") as fh:
        fh.write(html)

    active = sum(1 for t in topics if t["raw"] > 0)
    print(f"verified: 0/{len(topics)} heartCount mismatches")
    print(f"names: {len(leaks)} leak(s) in payload, "
          f"{len(forbidden)} name term(s) excluded from the search index")
    print(f"search: {len(payload['search']['index'])} terms, "
          f"{len(bodies)} bodies indexed")
    print(f"topics: {len(topics)} ({active} with >=1 heart)   "
          f"candidates: {len(electors)} shown, {dropped} zero-heart dropped")
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)} ({os.path.getsize(OUT_PATH) // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
