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


def main() -> int:
    with open(RAW_PATH) as fh:
        raw = json.load(fh)

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

    html = html.replace("/*__D3__*/", d3).replace("__DATA__", data)

    with open(OUT_PATH, "w") as fh:
        fh.write(html)

    active = sum(1 for t in topics if t["raw"] > 0)
    print(f"verified: 0/{len(topics)} heartCount mismatches")
    print(f"topics: {len(topics)} ({active} with >=1 heart)   "
          f"candidates: {len(electors)} shown, {dropped} zero-heart dropped")
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)} ({os.path.getsize(OUT_PATH) // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
