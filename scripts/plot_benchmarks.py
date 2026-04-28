"""
Collect all results and generate comparison figure + table.

Reads ONLY from local JSON files produced by models and kmers.

Produces:
    figures/retrieval_comparison.png  — grouped bar chart
    figures/map5_table.txt            — plain-text table for the report
    figures/all_results.json          — machine-readable merged results
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("figures", exist_ok=True)

OUTPUT_FOLDER = "results"
TASK_IDS      = ["arch_retrieval", "euk_retrieval"]
TASK_LABELS   = {"arch_retrieval": "Arch Retrieval", "euk_retrieval": "Euk Retrieval"}

# All AA foundation models in parameter order (matches DGEB paper ordering)
FOUNDATION_MODELS = [
    ("facebook/esm2_t6_8M_UR50D",    "ESM2-8M",        8,    "#4878CF"),
    ("facebook/esm2_t12_35M_UR50D",  "ESM2-35M",       35,   "#4878CF"),
    ("facebook/esm2_t30_150M_UR50D", "ESM2-150M",      150,  "#4878CF"),
    ("facebook/esm2_t33_650M_UR50D", "ESM2-650M",      650,  "#4878CF"),
    ("facebook/esm2_t36_3B_UR50D",   "ESM2-3B",        3000, "#4878CF"),
    ("hugohrban/progen2-small",       "ProGen2-S",      150,  "#6ACC65"),
    ("hugohrban/progen2-medium",      "ProGen2-M",      765,  "#6ACC65"),
    ("hugohrban/progen2-large",       "ProGen2-L",      2700, "#6ACC65"),
    ("hugohrban/progen2-xlarge",      "ProGen2-XL",     6400, "#6ACC65"),
    ("Rostlab/prot_t5_xl_uniref50",   "ProtTrans-U50",  1200, "#D65F5F"),
    ("Rostlab/prot_t5_xl_bfd",        "ProtTrans-BFD",  1200, "#D65F5F"),
]

K_VALUES = [1, 2, 3, 4, 5]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _extract_map5(data: dict, path: str) -> float:
    """
    Extract best MAP@5 across all layers from a loaded DGEB result dict.

    DGEB has produced at least two JSON structures across versions:

    Structure A (pydantic model_dump_json, newer):
        {"results": [{"layer_number": N, "metrics": [{"id": "map_at_5", "value": V}]}]}

    Structure B (from_dict / older serialization):
        {"layers": {"3": {"map_at_5": V, ...}, "5": {"map_at_5": V, ...}}}

    We try both and raise a clear error if neither works, printing the
    actual top-level keys so the format can be diagnosed.
    """
    # ── Structure A: results → list of LayerResult dicts ─────────────────────
    if "results" in data:
        layers = data["results"]
        scores = [
            m["value"]
            for layer in layers
            for m in layer.get("metrics", [])
            if m.get("id") == "map_at_5"
        ]
        if scores:
            return max(scores)

    # ── Structure B: layers → dict keyed by layer number ─────────────────────
    if "layers" in data:
        layers = data["layers"]
        scores = []
        for layer_key, metrics in layers.items():
            if isinstance(metrics, dict) and "map_at_5" in metrics:
                scores.append(metrics["map_at_5"])
        if scores:
            return max(scores)

    # ── Structure C: flat dict — map_at_5 is a direct top-level key ──────────
    # step3_kmer_baseline.py saves evaluator() output directly as JSON:
    # {"map_at_5": 0.017, "ndcg_at_5": 0.09, ...}  — no layer wrapper.
    if "map_at_5" in data:
        return float(data["map_at_5"])

    # ── None worked: print the file so the user can diagnose ─────────────
    import pprint
    print(f"\nERROR: Cannot extract map_at_5 from {path}")
    print(f"  Top-level keys: {list(data.keys())}")
    if "results" in data and data["results"]:
        print(f"  First 'results' entry keys: {list(data['results'][0].keys())}")
    if "layers" in data:
        first_key = next(iter(data["layers"]))
        print(f"  First 'layers' entry (key={first_key}): {data['layers'][first_key]}")
    print("  Full file contents (truncated):")
    pprint.pprint({k: v for k, v in data.items() if k != "results"}, depth=3)
    raise ValueError(f"Cannot parse map_at_5 from {path} — see output above.")


def load_map5(result_dir: str, task_id: str) -> float | None:
    """
    Load best MAP@5 (across layers) from a DGEB result JSON.
    Returns None if the file does not exist.
    """
    path = os.path.join(result_dir, f"{task_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return _extract_map5(data, path)


def load_all_metrics(result_dir: str, task_id: str) -> dict | None:
    """Load the full metrics dict (all layers, all metrics) from a result JSON."""
    path = os.path.join(result_dir, f"{task_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT FOUNDATION MODEL RESULTS
# ─────────────────────────────────────────────────────────────────────────────

fm_rows = []   # list of dicts with label, arch, euk, color, n_params

for hf_name, label, n_params_m, color in FOUNDATION_MODELS:
    short      = hf_name.split("/")[-1]
    result_dir = os.path.join(OUTPUT_FOLDER, short)
    arch       = load_map5(result_dir, "arch_retrieval")
    euk        = load_map5(result_dir, "euk_retrieval")
    if arch is not None or euk is not None:
        fm_rows.append({
            "label": label, "hf_name": hf_name,
            "n_params_m": n_params_m,
            "arch": arch, "euk": euk,
            "color": color,
        })

if not fm_rows:
    print("WARNING: No foundation model results found in results/. "
          "Run step2b_run_all_models.py first.")
else:
    print(f"Loaded {len(fm_rows)} foundation model result(s).")

# ─────────────────────────────────────────────────────────────────────────────
# COLLECT K-MER RESULTS
# ─────────────────────────────────────────────────────────────────────────────

kmer_rows = []

for k in K_VALUES:
    result_dir = os.path.join(OUTPUT_FOLDER, f"kmer_k{k}")
    arch = load_map5(result_dir, "arch_retrieval")
    euk  = load_map5(result_dir, "euk_retrieval")
    if arch is not None or euk is not None:
        kmer_rows.append({
            "label": f"k-mer (k={k})", "k": k,
            "arch": arch, "euk": euk,
            "color": "#F5A623",
        })

if not kmer_rows:
    print("WARNING: No k-mer results found in results/. "
          "Run step3_kmer_baseline.py first.")
else:
    print(f"Loaded {len(kmer_rows)} k-mer result(s).")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE MERGED JSON
# ─────────────────────────────────────────────────────────────────────────────

all_results = {
    "foundation_models": [
        {k: v for k, v in r.items() if k != "color"}
        for r in fm_rows
    ],
    "kmer_baselines": [
        {k: v for k, v in r.items() if k != "color"}
        for r in kmer_rows
    ],
}
with open("figures/all_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("Saved figures/all_results.json")

# ─────────────────────────────────────────────────────────────────────────────
# PLAIN-TEXT TABLE
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v):
    return f"{v:.5f}" if v is not None else "  N/A  "

lines = []
header = f"{'Method':<28} {'Arch Retrieval MAP@5':>22} {'Euk Retrieval MAP@5':>22}"
lines.append(header)
lines.append("-" * len(header))

lines.append("--- K-mer baselines (this work) ---")
for r in kmer_rows:
    lines.append(f"{r['label']:<28} {fmt(r['arch']):>22} {fmt(r['euk']):>22}")

lines.append("--- Foundation models (reproduced from DGEB) ---")
for r in fm_rows:
    tag = f"  [{r['n_params_m']}M]"
    lines.append(f"{r['label']:<28} {fmt(r['arch']):>22} {fmt(r['euk']):>22}{tag}")

table_str = "\n".join(lines)
print("\n" + table_str)

with open("figures/map5_table.txt", "w") as f:
    f.write(table_str + "\n")
print("\nSaved figures/map5_table.txt")

# ─────────────────────────────────────────────────────────────────────────────
# BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

# Combine: k-mer rows first, then foundation model rows
all_rows = kmer_rows + fm_rows

if not all_rows:
    print("No results to plot. Exiting.")
    raise SystemExit

labels     = [r["label"] for r in all_rows]
arch_vals  = [r["arch"]  if r["arch"]  is not None else float("nan") for r in all_rows]
euk_vals   = [r["euk"]   if r["euk"]   is not None else float("nan") for r in all_rows]
colors     = [r["color"] for r in all_rows]
n_kmer     = len(kmer_rows)

fig, axes = plt.subplots(1, 2, figsize=(max(12, len(all_rows) * 0.9), 5.5), sharey=False)
fig.suptitle(
    "MAP@5 on DGEB Retrieval Tasks: K-mer Baselines vs. Foundation Models\n"
    "(all results reproduced locally — no numbers from paper)",
    fontsize=12, fontweight="bold", y=1.02,
)

x     = np.arange(len(labels))
width = 0.65

for ax, vals, title in [
    (axes[0], arch_vals, "Arch Retrieval (MAP@5)"),
    (axes[1], euk_vals,  "Euk Retrieval (MAP@5)"),
]:
    bars = ax.bar(x, vals, width=width, color=colors, edgecolor="white", linewidth=0.6)

    # Hatch k-mer bars to distinguish visually
    for i, bar in enumerate(bars):
        if i < n_kmer:
            bar.set_hatch("///")
            bar.set_edgecolor("#333333")

    # Annotate bar tops
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    # Separator line between k-mer and FM groups
    if n_kmer > 0 and len(fm_rows) > 0:
        ax.axvline(x=n_kmer - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel("MAP@5", fontsize=10)
    ax.set_title(title, fontsize=11)
    ymax = max((v for v in vals if not np.isnan(v)), default=0.5)
    ax.set_ylim(0, ymax * 1.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

# Legend
kmer_patch = mpatches.Patch(
    facecolor="#F5A623", hatch="///", edgecolor="#333333", label="K-mer baseline (this work)"
)
legend_handles = [kmer_patch]
for color, family in [("#4878CF", "ESM2"), ("#6ACC65", "ProGen2"), ("#D65F5F", "ProtTrans")]:
    if any(r["color"] == color for r in fm_rows):
        legend_handles.append(mpatches.Patch(facecolor=color, label=f"{family} (reproduced)"))

fig.legend(
    handles=legend_handles, loc="lower center",
    bbox_to_anchor=(0.5, -0.12), ncol=len(legend_handles), fontsize=9,
)

plt.tight_layout()
fig.savefig("figures/retrieval_comparison.png", dpi=150, bbox_inches="tight")
print("Saved figures/retrieval_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# SCALING CURVE (MAP@5 vs. num_params for FM, reference lines for k-mer)
# ─────────────────────────────────────────────────────────────────────────────

if len(fm_rows) >= 2:
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig2.suptitle(
        "MAP@5 vs. Model Size — Scaling Behaviour\n"
        "K-mer baselines shown as horizontal reference lines",
        fontsize=12, fontweight="bold",
    )

    # Distinct color + style per k, light→dark so higher k is more salient
    kmer_palette = {
        "k-mer (k=1)": ("#FFCC80", "solid"),
        "k-mer (k=2)": ("#FFA726", "solid"),
        "k-mer (k=3)": ("#FB8C00", "dashed"),
        "k-mer (k=4)": ("#E65100", "dashed"),
        "k-mer (k=5)": ("#BF360C", "dashed"),
    }

    for ax, vals_key, title in [
        (axes2[0], "arch", "Arch Retrieval"),
        (axes2[1], "euk",  "Euk Retrieval"),
    ]:
        # Foundation model scaling lines
        for color, family, hf_prefix in [
            ("#4878CF", "ESM2",      "facebook/esm2"),
            ("#6ACC65", "ProGen2",   "hugohrban/progen2"),
            ("#D65F5F", "ProtTrans", "Rostlab/prot_t5"),
        ]:
            family_rows = [r for r in fm_rows if r["hf_name"].startswith(hf_prefix)]
            if not family_rows:
                continue
            pts = sorted(
                [(r["n_params_m"], r[vals_key]) for r in family_rows if r[vals_key] is not None]
            )
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", color=color, label=family,
                        linewidth=2.0, markersize=7, zorder=3)

        # K-mer reference lines — all k, labeled at right margin
        for row in kmer_rows:
            val = row[vals_key]
            if val is None:
                continue
            lbl = row["label"]
            color, ls = kmer_palette.get(lbl, ("#888888", "dashed"))
            lw = 1.8 if ("k=4" in lbl or "k=5" in lbl) else 1.2
            ax.axhline(val, linestyle=ls, color=color, linewidth=lw, alpha=0.9, zorder=2)
            ax.annotate(
                f"{lbl}  {val:.3f}",
                xy=(1.01, val),
                xycoords=("axes fraction", "data"),
                fontsize=7.5, color=color, va="center",
                fontweight="bold" if ("k=4" in lbl or "k=5" in lbl) else "normal",
            )

        ax.set_xscale("log")
        ax.set_xlabel("Number of parameters (millions, log scale)", fontsize=9)
        ax.set_ylabel("MAP@5", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    plt.tight_layout()
    fig2.subplots_adjust(right=0.82)
    fig2.savefig("figures/scaling_curve.png", dpi=150, bbox_inches="tight")
    print("Saved figures/scaling_curve.png")
