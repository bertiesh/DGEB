"""
Run all 12 AA foundation models on DGEB retrieval tasks.

Models evaluated (matching DGEB Appendix F, AA models only):
  ESM2    : esm2_t6_8M, esm2_t12_35M, esm2_t30_150M, esm2_t33_650M, esm2_t36_3B
  ProGen2 : progen2-small, progen2-medium, progen2-large, progen2-xlarge
  ProtTrans: prot_t5_xl_uniref50, prot_t5_xl_bfd
  ESM3    : esm3_sm_open_v1  (skipped by default — requires `pip install esm`)

RESUME SUPPORT:
  If a result JSON already exists for a model+task pair, that model is
  skipped. This means you can Ctrl+C at any time and re-run the script
  to continue where you left off. ESM2-8M was already run in step2, so
  it will be skipped automatically.

RUNTIME ESTIMATES (CPU, MacBook Pro M2):
  ESM2-8M      ~35 min (already done)
  ESM2-35M     ~40 min
  ESM2-150M    ~55 min
  ESM2-650M    ~90 min
  ESM2-3B      ~4 hr   ← skip with SKIP_LARGE=True below if time-constrained
  ProGen2-small   ~45 min
  ProGen2-medium  ~90 min
  ProGen2-large   ~3 hr  ← skip with SKIP_LARGE=True
  ProGen2-xlarge  ~6 hr  ← skip with SKIP_LARGE=True
  ProtTrans-uniref ~90 min
  ProtTrans-bfd    ~90 min

SET SKIP_LARGE=True TO RUN ONLY MODELS THAT FINISH IN REASONABLE TIME ON CPU.
Large models (>1B params) produce embeddings that are very close to medium
models on these tasks per Appendix F — skipping them loses little for the
baseline comparison while saving many hours of CPU time.
"""

import json
import os
import dgeb

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these to control which models run
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_FOLDER = "results"

# Set to True to skip models with >1B parameters.
# Recommended on a laptop CPU. Set to False on a GPU machine.
SKIP_LARGE = True

# All 11 standard AA models (ESM3 excluded — needs `pip install esm` separately)
# Format: (hf_model_name, num_params_millions, batch_size)
# batch_size is reduced for larger models to avoid RAM exhaustion.
ALL_MODELS = [
    # ESM2 series
    ("facebook/esm2_t6_8M_UR50D",    8,    32),
    ("facebook/esm2_t12_35M_UR50D",  35,   32),
    ("facebook/esm2_t30_150M_UR50D", 150,  16),
    ("facebook/esm2_t33_650M_UR50D", 650,  8),
    ("facebook/esm2_t36_3B_UR50D",   3000, 2),   # large — skipped if SKIP_LARGE
    # ProGen2 series
    ("hugohrban/progen2-small",   150,  16),
    ("hugohrban/progen2-medium",  765,  8),
    ("hugohrban/progen2-large",   2700, 2),       # large — skipped if SKIP_LARGE
    ("hugohrban/progen2-xlarge",  6400, 1),       # large — skipped if SKIP_LARGE
    # ProtTrans
    ("Rostlab/prot_t5_xl_uniref50", 1200, 4),    # large — skipped if SKIP_LARGE
    ("Rostlab/prot_t5_xl_bfd",      1200, 4),    # large — skipped if SKIP_LARGE
]

LARGE_THRESHOLD_M = 1000   # models above this many million params are "large"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def result_exists(model_name: str, task_id: str) -> bool:
    """Check whether a result JSON already exists for this model+task."""
    model_short = model_name.split("/")[-1]
    path = os.path.join(OUTPUT_FOLDER, model_short, f"{task_id}.json")
    return os.path.exists(path)


def all_results_exist(model_name: str, task_ids: list) -> bool:
    """Return True if all task results already exist for this model."""
    return all(result_exists(model_name, t) for t in task_ids)


def load_map5_from_json(model_name: str, task_id: str) -> float:
    """Read MAP@5 (best across layers) from a saved result JSON."""
    model_short = model_name.split("/")[-1]
    path = os.path.join(OUTPUT_FOLDER, model_short, f"{task_id}.json")
    with open(path) as f:
        data = json.load(f)
    # Structure: {"results": [{"layer_number": N, "metrics": [{"id": ..., "value": ...}]}]}
    best = max(
        m["value"]
        for layer in data["results"]
        for m in layer["metrics"]
        if m["id"] == "map_at_5"
    )
    return best


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    # Load retrieval tasks once
    all_protein_tasks = dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN)
    retrieval_tasks   = [t for t in all_protein_tasks if t.metadata.type == "retrieval"]
    task_ids          = [t.metadata.id for t in retrieval_tasks]

    print(f"Retrieval tasks: {[t.metadata.display_name for t in retrieval_tasks]}")
    print(f"SKIP_LARGE={SKIP_LARGE} (threshold: >{LARGE_THRESHOLD_M}M params)\n")

    skipped_large  = []
    skipped_cached = []
    ran            = []
    failed         = []

    for model_name, n_params_m, batch_size in ALL_MODELS:
        short = model_name.split("/")[-1]

        # Skip large models if requested
        if SKIP_LARGE and n_params_m >= LARGE_THRESHOLD_M:
            print(f"[SKIP-LARGE ] {short} ({n_params_m}M params)")
            skipped_large.append(model_name)
            continue

        # Skip if all results already cached
        if all_results_exist(model_name, task_ids):
            print(f"[CACHED     ] {short} — results already exist, skipping.")
            skipped_cached.append(model_name)
            continue

        print(f"\n{'='*60}")
        print(f"[RUNNING    ] {short} ({n_params_m}M params, batch_size={batch_size})")
        print(f"{'='*60}")

        try:
            model = dgeb.get_model(
                model_name=model_name,
                layers=None,
                devices=[0],
                max_seq_length=1024,
                batch_size=batch_size,
                pool_type="mean",
                num_processes=0,    # required on macOS/Windows (spawn)
            )
            evaluation = dgeb.DGEB(tasks=retrieval_tasks)
            evaluation.run(model, output_folder=OUTPUT_FOLDER)
            ran.append(model_name)
            print(f"[DONE       ] {short}")

        except Exception as e:
            print(f"[ERROR      ] {short}: {e}")
            failed.append((model_name, str(e)))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print(f"  Ran successfully : {len(ran)}")
    print(f"  Already cached   : {len(skipped_cached)}")
    print(f"  Skipped (large)  : {len(skipped_large)}")
    print(f"  Failed           : {len(failed)}")
    if failed:
        for name, err in failed:
            print(f"    {name}: {err}")

    # ── Print MAP@5 table for all available results ───────────────────────────
    print("\n" + "=" * 60)
    print("MAP@5 RESULTS (all models with cached results)")
    print("=" * 60)
    print(f"{'Model':<35} {'Arch MAP@5':>12} {'Euk MAP@5':>12}")
    print("-" * 60)

    for model_name, n_params_m, _ in ALL_MODELS:
        short = model_name.split("/")[-1]
        arch_ok = result_exists(model_name, "arch_retrieval")
        euk_ok  = result_exists(model_name, "euk_retrieval")
        if arch_ok and euk_ok:
            arch = load_map5_from_json(model_name, "arch_retrieval")
            euk  = load_map5_from_json(model_name, "euk_retrieval")
            print(f"{short:<35} {arch:>12.5f} {euk:>12.5f}")
        elif arch_ok or euk_ok:
            print(f"{short:<35}   (partial results)")
        else:
            tag = " [large, skipped]" if n_params_m >= LARGE_THRESHOLD_M and SKIP_LARGE else " [not yet run]"
            print(f"{short:<35}{tag}")


if __name__ == "__main__":
    run()