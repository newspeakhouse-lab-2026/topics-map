#!/usr/bin/env python3
"""Draw how the map's cluster composition moves as the min-hearts floor rises.

The cluster key in the map shows one bar per cluster *at the current floor*;
this script draws the same quantity for *every* floor as a single N-line figure,
so the trend is legible without dragging the slider:

    share(c, f) = # topics in cluster c with raw >= f
                  -------------------------------------
                        # topics with raw >= f

A cluster whose share climbs as the floor rises is one whose support is broad
(many candidates heart it); a cluster that collapses is niche. That is the
"relative size of clusters as a function of min multiplicity" view.

It reads the committed artefact (`map.html`) rather than the gitignored
intermediates, so the figure always matches what the map shows *and* can be
regenerated from a fresh clone without re-running the embedding stage. No name
ever reaches this file: the payload it reads has already passed build.py's
privacy guard, and cluster terms are stripped of name tokens by embed.py.

Usage:
    python3.11 -m pip install -r scripts/requirements-plot.txt
    python3.11 scripts/plot_clusters.py [--counts] [--out-dir figures] [--dpi 200]

Writes figures/cluster-share.png and figures/cluster-share.svg, or
figures/cluster-counts.* with --counts. Output is byte-identical across runs:
the timestamps matplotlib would otherwise stamp into PNG/SVG are dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_PATH = os.path.join(ROOT, "map.html")
OUT_DIR = os.path.join(ROOT, "figures")

# build.py escapes every "<" in the payload, so the first </script> after the
# tag really is the end of it.
PAYLOAD_RE = re.compile(
    r'<script id="payload" type="application/json">(.*?)</script>', re.S)

# The map colours cluster i with d3.schemeTableau10[i % 10]; the same list here
# is what makes the figure and the legend agree by construction rather than by
# coincidence. Hardcoded because this stage must not depend on reading d3.
TABLEAU10 = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f",
             "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab"]

INK, MUTED, LINE = "#1b2330", "#6b7385", "#dfe3ec"


def load_payload(path):
    """The inlined payload of the built map -- already verified and name-free."""
    with open(path) as fh:
        html = fh.read()
    match = PAYLOAD_RE.search(html)
    if not match:
        raise SystemExit(f"ERROR: no payload script tag in {path} -- run "
                         "scripts/build.py (or point --html at a built map).")
    data = json.loads(match.group(1))
    if not data.get("semantic"):
        raise SystemExit(
            "ERROR: that build carries no semantic layer, so there are no clusters "
            "to plot. data/embeddings.json was missing when it was built: run "
            "scripts/embed.py, then scripts/build.py.")
    return data


def sweep(data):
    """Per floor: how many topics survive, and how they split across clusters.

    Floors run 0..max(raw), the whole range the map's slider can express at the
    top of the axis; floor 0 keeps everything, including topics nobody hearted.
    """
    semantic = data["semantic"]
    labels = semantic["clusters"]
    topics = data["topics"]
    if len(labels) != len(topics):
        raise SystemExit("ERROR: payload is inconsistent (cluster count != topic "
                         "count); rebuild with scripts/build.py.")
    k = len(semantic["clusterTerms"])

    floors, rows = [], []
    top = max([t["raw"] for t in topics] or [0])
    for floor in range(top + 1):
        kept = [c for c, t in zip(labels, topics) if t["raw"] >= floor]
        counts = [0] * k
        for c in kept:
            counts[c] += 1
        floors.append(floor)
        rows.append((len(kept), counts))
    return floors, rows, k, [list(t) for t in semantic["clusterTerms"]]


def report(floors, rows, k, terms, as_counts):
    """A terminal table, so the numbers are quotable without opening the PNG."""
    head = "floor  topics  " + "  ".join(f"c{c:<6}" for c in range(k))
    print(head)
    print("-" * len(head))
    for floor, (n, counts) in zip(floors, rows):
        cells = []
        for c in range(k):
            cells.append(f"{counts[c]:<7}" if as_counts
                         else f"{100*counts[c]/n:5.1f}% " if n else "   n/a ")
        print(f"{floor:>5}  {n:>6}  " + "  ".join(cells))
    print()
    if not as_counts:
        for c in range(k):
            base = 100 * rows[0][1][c] / rows[0][0]
            peak = max(range(len(rows)), key=lambda f: (rows[f][1][c] / rows[f][0], -f))
            gone = f", empty by floor {floors[-1]}" if peak and not rows[-1][1][c] else ""
            print(f"  c{c} {', '.join(terms[c][:3]):<40} "
                  f"floor 0 {base:5.1f}%  ->  peak {100*rows[peak][1][c]/rows[peak][0]:5.1f}% "
                  f"at floor {floors[peak]}" + gone)


def render(floors, rows, k, terms, out_dir, *, as_counts, dpi):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(f"ERROR: {exc}\nInstall the plot stage:\n"
                         "  python3.11 -m pip install -r scripts/requirements-plot.txt")

    # Keeps the emitted SVG free of a creation date, so the file only changes
    # when the data does (same promise the rest of the pipeline makes).
    matplotlib.rcParams["svg.hashsalt"] = "faculty-topic-map"

    fig, ax = plt.subplots(figsize=(9.8, 5.7))
    # Generous bottom: a row of "how many topics are left" sits under the ticks,
    # with a two-line footnote below everything.
    fig.subplots_adjust(left=0.085, right=0.665, top=0.79, bottom=0.27)

    series = []
    for c in range(k):
        if as_counts:
            series.append([row[1][c] for row in rows])
        else:
            series.append([100 * row[1][c] / row[0] if row[0] else 0.0 for row in rows])

    for c in range(k):
        ax.plot(floors, series[c], marker="o", markersize=4.2, linewidth=2,
                color=TABLEAU10[c % len(TABLEAU10)], clip_on=False, zorder=3)

    top = max(max(s) for s in series) if series else 1
    ax.set_ylim(0, top * 1.14)
    if as_counts:
        ax.set_ylabel("topics in cluster (count)", fontsize=9.5, color=MUTED)
    else:
        ax.set_ylabel("share of the topics on the map", fontsize=9.5, color=MUTED)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")

    ax.set_xticks(floors)
    ax.set_xlabel("minimum hearts  (raw \u2764)", fontsize=9.5, color=MUTED, labelpad=20)
    ax.grid(axis="y", color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)

    # How many topics each floor leaves, directly under its tick: a 37% share of
    # 19 topics and a 37% share of 150 are the same number and not the same
    # claim, so the denominator travels with the axis. Far up the range it gets
    # too small for a share to mean anything, so those counts are flagged in the
    # map's candidate orange.
    SMALL = 10
    for floor, (n, _) in zip(floors, rows):
        ax.text(floor, -0.105, f"{n}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8,
                color="#e8590c" if n < SMALL else MUTED)

    cluster_word = "clusters" if k != 1 else "cluster"
    fig.text(0.085, 0.958, "Cluster share as a function of minimum hearts",
             fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.085, 0.908,
             f"share of the topics at or above each heart floor, by {cluster_word} "
             f"(k-means, k={k}) · numbers under the axis: topics kept",
             fontsize=9.6, color=MUTED, va="top")

    handles = [plt.Line2D([], [], color=TABLEAU10[c % len(TABLEAU10)], lw=2,
                          marker="o", markersize=4.2,
                          label=f"{c}  " + ", ".join(terms[c][:3]))
               for c in range(k)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.025, 1.0),
              frameon=False, fontsize=8.8, handlelength=1.7, labelspacing=0.72,
              title="cluster · top c-TF-IDF terms", title_fontsize=8.8,
              borderaxespad=0)

    fig.text(0.085, 0.06,
             "share = topics in cluster / topics kept · clusters are k-means in the full "
             "384-d title+body embedding\n(all-MiniLM-L6-v2; k chosen by silhouette, 0.036), "
             "not on the 2-D map, so same-coloured topics are not always neighbours.",
             fontsize=7.6, color=MUTED, linespacing=1.55, va="bottom")

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "cluster-counts" if as_counts else "cluster-share")
    # metadata drops the "Software"/"Date" stamps that would otherwise make two
    # runs over unchanged data produce different bytes.
    fig.savefig(stem + ".png", dpi=dpi, metadata={"Software": None})
    fig.savefig(stem + ".svg", metadata={"Date": None})
    plt.close(fig)
    return stem + ".png", stem + ".svg"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--html", default=HTML_PATH,
                    help="built map to read the payload from (default: map.html)")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="directory for the figure (default: figures/)")
    ap.add_argument("--counts", action="store_true",
                    help="plot absolute cluster sizes instead of shares")
    ap.add_argument("--dpi", type=int, default=200, help="PNG resolution (default: 200)")
    args = ap.parse_args()

    data = load_payload(args.html)
    floors, rows, k, terms = sweep(data)
    print(f"{os.path.relpath(args.html, ROOT)}: {len(data['topics'])} topics, "
          f"k={k} clusters, floors 0..{floors[-1]}\n")
    report(floors, rows, k, terms, args.counts)
    png, svg = render(floors, rows, k, terms, args.out_dir,
                      as_counts=args.counts, dpi=args.dpi)
    for path in (png, svg):
        print(f"wrote {os.path.relpath(path, ROOT)} ({os.path.getsize(path) // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
