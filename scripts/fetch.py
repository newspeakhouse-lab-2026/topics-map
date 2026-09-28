#!/usr/bin/env python3
"""Fetch the Newspeak House 2026-27 topic/heart graph from topic.forum.

Public data only. No authentication is required: the forum is readable
anonymously, and `topicFeed(heartedBy: <electorId>)` exposes each Fellowship
Candidate's hearts without a token.

Two things shape the request pattern:

  * `limit` is capped at 50 for both `topicFeed` and `heartedBy`, and the cap
    is silent -- you get 50 rows with no indication there are more. So every
    list has to be offset-paginated.
  * Queries are batched with GraphQL aliases, so one HTTP request carries a
    page for every candidate at once. This is a small community-run forum:
    batching keeps a full refresh at ~4 requests instead of ~30, which is
    kinder to the server, avoids rate limiting, and reads closer to a single
    consistent snapshot.

Responses are cached under data/cache/ so re-runs are free. Use --refresh to
bypass the cache (the forum resets hearts termly, so a scheduled re-run is
expected).

Privacy: candidate full names are used only to derive initials and are then
discarded, and Clerk user ids are replaced by a short deterministic digest
(see candidate_token). data/raw.json (tracked) carries token + initials,
never a name, person slug or raw user id. The cache under data/cache/ holds
raw API responses (which do contain names) and is gitignored -- do not commit
it.

Topic bodies are fetched too: the semantic search / embedding work needs real
prose. Bodies routinely name members, so they are normalised to plain text and
written to data/bodies.json, which is *also* gitignored -- body text must never
reach a tracked file. fetch.py additionally drops a data/cache/names.json name
list so build.py can prove no name, slug or name token leaked into the artefact.

Usage:
    python3 scripts/fetch.py [--refresh]

Writes data/raw.json (tracked); data/bodies.json and data/cache/names.json
(gitignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

ENDPOINT = "https://topic.forum/graphql"
SLUG = "newspeak-house-2026-27"
# Identify the client so the forum's operators can see (and contact us about)
# this traffic instead of guessing at an anonymous urllib user agent.
USER_AGENT = "faculty-topic-map/1.0 (+https://github.com/mrmvn/faculty-topic-map)"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for `import build`
import build  # noqa: E402  (stdlib-only; shared semantic_key / content hash)

CACHE_DIR = os.path.join(ROOT, "data", "cache")
RAW_PATH = os.path.join(ROOT, "data", "raw.json")
# Body text + the name list are gitignored: both routinely name members and
# must never reach a tracked file.
BODIES_PATH = os.path.join(ROOT, "data", "bodies.json")
NAMES_PATH = os.path.join(CACHE_DIR, "names.json")

PAGE = 50  # server-enforced maximum per query
# Pages packed into a single HTTP request via GraphQL aliases. Batching keeps a
# full refresh to a handful of requests rather than one per candidate per page.
PAGES_PER_REQUEST = 4

TOPIC_FIELDS = "id title slug heartCount publishedAt bodyMd"

_MD_FENCE = re.compile(r"```.*?```", re.S)
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_URL = re.compile(r"https?://\S+")
_MD_TAG = re.compile(r"<[^>]+>")
_MD_MARKS = re.compile(r"[*_`>#~]+")


def plain_text(md: str) -> str:
    """Reduce topic markdown to plain prose.

    Links keep their text but lose their href (a href can carry a person slug);
    code fences, images, raw URLs and emphasis markers are dropped. This is the
    form the embedding stage consumes, and it deliberately carries no URLs.
    """
    text = _MD_FENCE.sub(" ", md)
    text = _MD_IMAGE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_URL.sub(" ", text)
    text = _MD_TAG.sub(" ", text)
    text = _MD_MARKS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


class FetchError(RuntimeError):
    pass


TOKEN_LEN = 10


def candidate_token(user_id: str) -> str:
    """Opaque, stable token standing in for a candidate's Clerk user id.

    Deterministic on purpose: a candidate keeps the same token across terms,
    which is what makes the termly snapshots comparable. Token length is fixed
    so the artefact never carries a resolvable id.

    Honest limit: `forumPeople` publicly returns userId + name, so anyone can
    recompute these tokens for every member and match them. This defeats casual
    reading of the artefact (no names, no user ids) but is NOT protection
    against someone who knows the source is public. A keyed HMAC with a local
    secret would be needed for that, at the cost of reproducibility.
    """
    return hashlib.sha256(user_id.encode()).hexdigest()[:TOKEN_LEN]


def make_initials(people: list[dict]) -> dict[str, str]:
    """Map userId -> unique initials.

    "Ada Lovelace" -> "AL", a mononym -> its single initial. Collisions are
    resolved by a stable numeric suffix ordered by userId, so the mapping does
    not depend on input order. Full names never leave this function.
    """

    def base(name: str) -> str:
        parts = [p for p in re.split(r"[\s\-'\u2019]+", name.strip()) if p]
        if not parts:
            return "?"
        if len(parts) == 1:
            return parts[0][0].upper()
        return "".join(p[0].upper() for p in parts)

    initials = {p["userId"]: base(p["name"]) for p in people}
    groups: dict[str, list[str]] = {}
    for uid, value in initials.items():
        groups.setdefault(value, []).append(uid)
    for value, uids in groups.items():
        if len(uids) > 1:
            for n, uid in enumerate(sorted(uids), 1):
                initials[uid] = f"{value}{n}"
    return initials


def gql(query: str, cache_key: str, refresh: bool) -> dict:
    """POST one GraphQL document, memoised to data/cache/<cache_key>.json.

    The cache file name includes a digest of the query text, so editing a
    query (e.g. adding a field) invalidates its stale cached response instead
    of silently reusing a payload that lacks the new field.
    """
    digest = hashlib.sha256(query.encode()).hexdigest()[:8]
    path = os.path.join(CACHE_DIR, f"{cache_key}-{digest}.json")
    if not refresh and os.path.exists(path) and os.path.getsize(path) > 2:
        with open(path) as fh:
            return json.load(fh)

    body = json.dumps({"query": query}).encode()
    last: str = "unknown"
    for attempt in range(5):
        try:
            req = urllib.request.Request(
                ENDPOINT,
                data=body,
                headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                text = resp.read().decode()
            if not text.strip():
                last = "empty response body"
            else:
                payload = json.loads(text)
                if "errors" in payload:
                    raise FetchError(f"GraphQL errors: {payload['errors']}")
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(path, "w") as fh:
                    json.dump(payload, fh)
                return payload
        except (urllib.error.URLError, FetchError, json.JSONDecodeError, TimeoutError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2.5 * (attempt + 1))
    raise FetchError(f"{cache_key}: giving up after 5 attempts ({last})")


def fetch_people(refresh: bool) -> list[dict]:
    payload = gql(
        '{ forumPeople(idOrSlug: "%s") { userId slug name roles } }' % SLUG,
        "people",
        refresh,
    )
    people = payload["data"]["forumPeople"]
    if not people:
        raise FetchError("forumPeople returned nothing")
    return people


def fetch_topics(refresh: bool) -> tuple[dict[str, dict], dict[str, str]]:
    """All published topics, via offset pagination (limit is capped at 50).

    Returns (topics, bodies): `topics` is the public record with body text
    already popped off, `bodies` maps topic id -> normalised plain text. The
    two are kept apart on purpose, so a body can never leak into raw.json by
    accident.
    """
    topics: dict[str, dict] = {}
    bodies: dict[str, str] = {}
    offset = 0
    while True:
        parts = [
            't%d: topicFeed(idOrSlug: "%s", limit: %d, offset: %d) { %s }'
            % (i, SLUG, PAGE, offset + i * PAGE, TOPIC_FIELDS)
            for i in range(PAGES_PER_REQUEST)
        ]
        payload = gql("query {\n" + "\n".join(parts) + "\n}", f"topics_{offset}", refresh)
        got = 0
        for rows in payload["data"].values():
            for topic in rows:
                body = topic.pop("bodyMd", None)
                topics[topic["id"]] = topic
                bodies[topic["id"]] = plain_text(body or "")
                got += 1
        print(f"  topics offset {offset}: +{got} (total {len(topics)})")
        # A short page means we have reached the end of the list.
        if got < PAGES_PER_REQUEST * PAGE:
            break
        offset += PAGES_PER_REQUEST * PAGE
    return topics, bodies


def fetch_hearts(electors: list[dict], refresh: bool) -> dict[str, list[str]]:
    """For every elector, the ids of topics they hearted (offset-paginated)."""
    hearts: dict[str, list[str]] = {e["slug"]: [] for e in electors}
    offset = 0
    while True:
        parts = [
            'e%d: topicFeed(idOrSlug: "%s", heartedBy: "%s", limit: %d, offset: %d) { id }'
            % (i, SLUG, e["userId"], PAGE, offset)
            for i, e in enumerate(electors)
        ]
        payload = gql("query {\n" + "\n".join(parts) + "\n}", f"hearts_{offset}", refresh)
        got = full = 0
        for i, elector in enumerate(electors):
            ids = [row["id"] for row in payload["data"].get(f"e{i}", [])]
            hearts[elector["slug"]].extend(ids)
            got += len(ids)
            if len(ids) == PAGE:
                full += 1
        print(f"  hearts offset {offset}: +{got} (total {sum(len(v) for v in hearts.values())})")
        # Stop only once *no* elector filled a page. Comparing the aggregate
        # against len(electors)*PAGE is wrong: a handful of prolific hearters
        # can still have pages while most candidates return short ones.
        if full == 0:
            break
        offset += PAGE
    return hearts


def load_json(path: str):
    """Read a JSON file, or None if it is missing or unreadable."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def report_changes(old, old_bodies, dataset, bodies) -> str:
    """Compare this fetch with the baseline on disk and say what must run.

    Returns ``(outcome, summary)``; outcome is one of three:

      "none"       nothing moved -- embeddings and map are both current
      "hearts"     only edges changed -- rebuild; embeddings still valid
      "structural" topic set / title / body changed -- embeddings are stale

    Hearts move daily, embedding inputs rarely, and re-embedding is the stage
    with third-party dependencies, so telling the two apart is what keeps a
    scheduled refresh cheap. The decisive signal is `semanticKey`, the
    body-inclusive content hash stored in raw.json: it catches a title or body
    edit that leaves the topic id set unchanged, which an id-only check misses.
    The per-field diff below is for humans; the hash decides.

    Prints a machine-readable `change=<outcome>` line to $GITHUB_OUTPUT when set,
    so a workflow can branch on it without parsing prose.
    """
    if old is None:
        print("\nbaseline: none (data/raw.json absent) -- initial fetch")
        return "structural", "initial fetch"

    old_topics = {t["id"]: t for t in old.get("topics", [])}
    new_topics = {t["id"]: t for t in dataset["topics"]}
    shared = old_topics.keys() & new_topics.keys()
    added = new_topics.keys() - old_topics.keys()
    removed = old_topics.keys() - new_topics.keys()
    retitled = [i for i in shared if old_topics[i]["title"] != new_topics[i]["title"]]
    body_edits = (
        [] if old_bodies is None
        else [i for i in shared if old_bodies.get(i, "") != bodies.get(i, "")]
    )

    old_edges = sum(len(e["heartedTopicIds"]) for e in old.get("electors", []))
    new_edges = sum(len(e["heartedTopicIds"]) for e in dataset["electors"])

    old_key = old.get("semanticKey")
    key_changed = old_key is not None and old_key != dataset["semanticKey"]

    print(f"\nchanges vs baseline ({old.get('generatedAt', '?')}):")
    print(f"  topics: {len(old_topics)} -> {len(new_topics)} "
          f"(+{len(added)} new, -{len(removed)} removed)")
    note = "" if old_bodies is not None else " (bodies unavailable to compare)"
    print(f"  edits:  {len(retitled)} retitled, {len(body_edits)} body edits{note}")
    print(f"  hearts: {old_edges} -> {new_edges} ({new_edges - old_edges:+d})")
    if old_key is None:
        print("  note: baseline predates semanticKey; a body-only edit could go "
              "undetected this once (next fetch stores the hash)")
    elif key_changed:
        print(f"  semanticKey: {old_key} -> {dataset['semanticKey']}")

    if added or removed or retitled or body_edits or key_changed:
        print("  => STRUCTURAL: embedding inputs changed; re-embed, then rebuild")
        print("     python3.11 scripts/embed.py && python3 scripts/build.py")
        return "structural", _summary(added, removed, retitled, body_edits,
                                      new_edges - old_edges)
    if new_edges != old_edges:
        print("  => hearts only: rebuild (embeddings still valid)")
        print("     python3 scripts/build.py")
        return "hearts", _summary(added, removed, retitled, body_edits,
                                  new_edges - old_edges)
    print("  => no change; nothing to do")
    return "none", "no change"


def _summary(added, removed, retitled, body_edits, heart_delta) -> str:
    """One-line human summary for the commit message."""
    parts = []
    if added:
        parts.append(f"+{len(added)} topics")
    if removed:
        parts.append(f"-{len(removed)} topics")
    if retitled:
        parts.append(f"{len(retitled)} retitled")
    if body_edits:
        parts.append(f"{len(body_edits)} body edits")
    if heart_delta:
        parts.append(f"{heart_delta:+d} hearts")
    return ", ".join(parts) or "no change"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-fetch")
    args = ap.parse_args()

    print(f"forum: {SLUG}  endpoint: {ENDPOINT}")
    people = fetch_people(args.refresh)
    electors = [p for p in people if "elector" in p["roles"]]
    # Names are used only to derive initials, then dropped: raw.json is a
    # tracked artefact and must not carry identifying names.
    initials = make_initials(electors)
    label = {e["slug"]: initials[e["userId"]] for e in electors}
    print(f"people: {len(people)} ({len(electors)} Fellowship Candidates)")

    topics, bodies = fetch_topics(args.refresh)
    hearts = fetch_hearts(electors, args.refresh)

    # Name list for build.py's leak guard. Covers every forum member, not just
    # candidates: a body may name a host or guest just as easily.
    with open(NAMES_PATH, "w") as fh:
        json.dump(
            {
                "names": sorted({p["name"] for p in people if p.get("name")}),
                "slugs": sorted({p["slug"] for p in people if p.get("slug")}),
                "tokens": sorted(
                    {
                        t.lower()
                        for p in people
                        if p.get("name")
                        for t in re.split(r"[\s\-'\u2019]+", p["name"])
                        if len(t) >= 4
                    }
                ),
            },
            fh,
            indent=1,
        )

    published = set(topics)
    # The app's weight denominator counts *published* topics only; a heart on a
    # since-unpublished topic is a stale row and must not dilute the weight.
    for slug, ids in hearts.items():
        stale = [i for i in ids if i not in published]
        if stale:
            print(f"  note: {label[slug]} has {len(stale)} heart(s) on unpublished topics (dropped)")

    dataset = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "forum": {"slug": SLUG, "endpoint": ENDPOINT},
        "electors": [
            {
                "id": candidate_token(e["userId"]),
                "initials": initials[e["userId"]],
                "heartedTopicIds": [i for i in hearts[e["slug"]] if i in published],
            }
            for e in electors
        ],
        "topics": sorted(topics.values(), key=lambda t: t.get("publishedAt") or ""),
        # Content hash of the embedding inputs (id + title + body). Lets
        # build.py prove embeddings.json is not stale, and lets the change
        # report tell a heart-only day (rebuild) from a topic/body edit
        # (re-embed). Bodies are hashed, never stored -- this field is a digest.
        "semanticKey": build.semantic_key(topics.values(), bodies),
    }

    edges = sum(len(e["heartedTopicIds"]) for e in dataset["electors"])
    heart_total = sum(t.get("heartCount") or 0 for t in dataset["topics"])
    print(f"\ntopics: {len(dataset['topics'])}   edges: {edges}   sum(heartCount): {heart_total}")
    if edges != heart_total:
        print(
            "  WARNING: recovered edges != sum(heartCount); the edge list may be "
            "incomplete (check for a per-page cap we missed).",
            file=sys.stderr,
        )

    # --- change report vs the baseline on disk ---------------------------
    # Loaded before the writes below so the gate sees the previous snapshot.
    old_raw = load_json(RAW_PATH)
    old_bodies = load_json(BODIES_PATH)
    change, summary = report_changes(old_raw, old_bodies, dataset, bodies)
    # A no-op fetch must not churn the artefact: keep the old timestamp so
    # raw.json -- and therefore map.html -- stays byte-identical when nothing
    # moved, and the workflow's "commit only on diff" check stays quiet.
    if change == "none" and old_raw and old_raw.get("generatedAt"):
        dataset["generatedAt"] = old_raw["generatedAt"]
    if os.environ.get("GITHUB_OUTPUT"):
        # Consumed by .github/workflows/refresh.yml: `change` branches the job,
        # `semanticKey` is the embeddings cache key (so heart-only days reuse a
        # cached embed), `summary` goes into the commit message.
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"change={change}\n")
            fh.write(f"semanticKey={dataset['semanticKey']}\n")
            fh.write(f"summary={summary}\n")

    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    with open(RAW_PATH, "w") as fh:
        json.dump(dataset, fh, indent=1)
    with open(BODIES_PATH, "w") as fh:
        json.dump(
            {t["id"]: bodies.get(t["id"], "") for t in dataset["topics"]},
            fh,
            indent=1,
        )

    body_words = [len(b.split()) for b in bodies.values()]
    empty = sum(1 for n in body_words if not n)
    if body_words:
        print(f"bodies: {len(bodies)} topics, {empty} empty, "
              f"{sum(body_words)} words (median {sorted(body_words)[len(body_words) // 2]})")

    print(f"wrote {os.path.relpath(RAW_PATH, ROOT)} (tracked)")
    print(f"wrote {os.path.relpath(BODIES_PATH, ROOT)} (gitignored, contains names)")
    print(f"wrote {os.path.relpath(NAMES_PATH, ROOT)} (gitignored, name guard)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
