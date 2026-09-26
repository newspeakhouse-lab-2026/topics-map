#!/usr/bin/env python3
"""Embed topics and project them to 2D for the semantic view.

This is the one stage that needs third-party packages; `build.py` stays
stdlib-only and offline. It reads the normalised bodies from
`data/bodies.json` (so topics are embedded on title + body, not title alone),
embeds them with a pinned sentence-transformer, and writes:

  data/embeddings.json (gitignored) --
      ids, 2D coords, kNN neighbour lists, k-means clusters + c-TF-IDF terms,
      and int8-quantised vectors, all keyed to data/raw.json's topic order.

`build.py` inlines the compact parts (coords / knn / clusters / terms) into
map.html; the vectors stay here for future use.

Setup (the environment here has no working venv and blocks TLS to PyPI, so
deps go into a project-local --target dir that this script puts on sys.path):

    python3.11 -m pip install --target .embed-deps --trusted-host pypi.org \\
        --trusted-host files.pythonhosted.org fastembed scikit-learn
    python3.11 scripts/embed.py

Why 3.11 and not the system 3.14: torch/onnxruntime publish no cp314 wheels.
fastembed (ONNX Runtime) is used instead of sentence-transformers/torch because
it is ~10x smaller and installs cleanly. Model weights are cached under
data/cache/ (gitignored), and the HF cache is redirected there too: the sandbox
denies writes to ~/.cache and blocks the Xet transfer protocol.

Usage:
    python3.11 scripts/embed.py

Writes data/embeddings.json.
"""

from __future__ import annotations

import base64
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for `import build`
DEPS = os.path.join(ROOT, ".embed-deps")

RAW_PATH = os.path.join(ROOT, "data", "raw.json")
BODIES_PATH = os.path.join(ROOT, "data", "bodies.json")
OUT_PATH = os.path.join(ROOT, "data", "embeddings.json")
CACHE_DIR = os.path.join(ROOT, "data", "cache")

# Local --target install goes on sys.path before the heavy imports below.
if os.path.isdir(DEPS):
    sys.path.insert(0, DEPS)

# Keep every cache inside the repo: ~/.cache is not writable here, and the
# Xet protocol endpoint is blocked, so force plain HTTP downloads.
os.environ.setdefault("HF_HOME", os.path.join(CACHE_DIR, "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(CACHE_DIR, "hf", "hub"))
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# joblib barfs reading physical-core count under the sandbox; one core is plenty.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import build  # noqa: E402  (stdlib-only; reuses analyze/drop_terms/load_names)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
NEIGHBOURS = 6
MAX_TERMS = 10
CLUSTER_RANGE = range(4, 11)


def l2_normalise(vectors):
    import numpy as np

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)


def project_2d(unit):
    """t-SNE to 2D. Deterministic (fixed seed, PCA init, exact gradients)."""
    from sklearn.manifold import TSNE

    perplexity = min(15.0, float(len(unit) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=SEED,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(unit), perplexity


def choose_clusters(unit):
    """k-means at the k with the best silhouette score -- no magic constant."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best = None
    for k in CLUSTER_RANGE:
        if k >= len(unit):
            break
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(unit)
        score = silhouette_score(unit, labels)
        if best is None or score > best[0]:
            best = (score, k, labels)
    score, k, labels = best
    return list(labels), k, score


def cluster_terms(texts, labels, k, forbidden):
    """Top c-TF-IDF terms per cluster.

    Name tokens are dropped (`forbidden`): a cluster keyword that is a member's
    name would both be useless and trip build.py's privacy guard.
    """
    tf = defaultdict(Counter)
    clusters_with_term = defaultdict(set)
    for i, label in enumerate(labels):
        counts = Counter(t for t in build.analyze(texts[i]) if t not in forbidden)
        tf[label].update(counts)
        for term in counts:
            clusters_with_term[term].add(label)

    terms = []
    for label in range(k):
        total = sum(tf[label].values()) or 1
        scored = [
            ((n / total) * math.log(1 + k / len(clusters_with_term[term])), term)
            for term, n in tf[label].items()
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        terms.append([term for _, term in scored[:MAX_TERMS]])
    return terms


def neighbours(unit, count):
    import numpy as np

    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -1.0)
    order = np.argsort(-similarity, axis=1)[:, :count]
    scores = np.take_along_axis(similarity, order, axis=1)
    return order.tolist(), [[round(float(s), 4) for s in row] for row in scores]


def quantise_int8(vectors):
    """Per-vector int8 quantisation; returns (bytes, scale per vector)."""
    import numpy as np

    scale = np.max(np.abs(vectors), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    quantised = np.round(vectors / scale * 127.0).astype(np.int8)
    return quantised.tobytes(), scale.ravel().tolist()


def main() -> int:
    try:
        import numpy as np  # noqa: F401
        from fastembed import TextEmbedding
    except ImportError as exc:
        print(f"ERROR: {exc}\nInstall the embed stage (see this file's docstring):\n"
              "  python3.11 -m pip install --target .embed-deps "
              "--trusted-host pypi.org --trusted-host files.pythonhosted.org "
              "fastembed scikit-learn", file=sys.stderr)
        return 1

    with open(RAW_PATH) as fh:
        topics = json.load(fh)["topics"]
    try:
        with open(BODIES_PATH) as fh:
            bodies = json.load(fh)
    except FileNotFoundError:
        print("ERROR: data/bodies.json missing -- run scripts/fetch.py first "
              "(embeddings need the body text, not just titles).", file=sys.stderr)
        return 1

    texts = [t["title"] + "\n" + bodies.get(t["id"], "") for t in topics]
    print(f"embedding {len(texts)} topics with {MODEL} ...")
    model = TextEmbedding(MODEL, cache_dir=os.path.join(CACHE_DIR, "fastembed"))
    vectors = np.array(list(model.embed(texts)), dtype=np.float32)
    unit = l2_normalise(vectors)
    print(f"  vectors: {vectors.shape[0]} x {vectors.shape[1]}")

    coords, perplexity = project_2d(unit)
    labels, k, silhouette = choose_clusters(unit)
    forbidden = build.drop_terms(build.load_names())
    terms = cluster_terms(texts, labels, k, forbidden)
    knn, knn_scores = neighbours(unit, NEIGHBOURS)
    blob, scales = quantise_int8(vectors)

    payload = {
        "model": MODEL,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dim": int(vectors.shape[1]),
        "projection": {"method": "tsne", "perplexity": perplexity, "seed": SEED},
        "clustering": {"method": "kmeans", "k": k, "seed": SEED,
                       "silhouette": round(silhouette, 4)},
        "ids": [t["id"] for t in topics],
        "coords": [[round(float(x), 3), round(float(y), 3)] for x, y in coords],
        "clusters": [int(c) for c in labels],
        "clusterTerms": terms,
        "knn": knn,
        "knnScores": knn_scores,
        "vectors": {
            "encoding": "int8-base64",
            "scales": [round(s, 6) for s in scales],
            "data": base64.b64encode(blob).decode(),
        },
    }

    with open(OUT_PATH, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    print(f"  tsne: perplexity {perplexity}, {len(coords)} points")
    print(f"  clusters: k={k} (silhouette {silhouette:.3f})")
    for i, group in enumerate(terms):
        size = sum(1 for c in labels if c == i)
        print(f"    {i} ({size:>2}): {', '.join(group[:6])}")
    print(f"  knn: {NEIGHBOURS} neighbours/topic, "
          f"first: {[topics[j]['title'][:28] for j in knn[0][:3]]}")
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)} "
          f"({os.path.getsize(OUT_PATH) // 1024} KB, gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
