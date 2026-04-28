"""
K-mer frequency baseline for DGEB retrieval tasks.

NOTE on qrels structure:
  The Arch Retrieval qrels have 163,612 rows across 2,343 queries,
  meaning ~69.83 relevant corpus entries per query on average.
  Relevance label is 1 (fuzz_ratio > 90 during dataset construction),
  stored as fuzz_ratio=1 in the HuggingFace dataset.
  pytrec_eval treats any score > 0 as relevant, so MAP@5 measures
  whether any of the ~70 relevant bacterial proteins appear in the
  top-5 retrieved results.

We implement k-mer frequency vectors and evaluate k in {1, 2, 3, 4, 5}.

MEMORY STRATEGY:
  k <= 4: vocab size <= 160,000 → dense float32 matrix, pass to RetrievalEvaluator.
  k >= 5: vocab size >= 3,200,000 → scipy sparse CSR matrix to avoid OOM.
          Cosine similarity computed as sparse @ sparse.T, then pytrec_eval
          called directly (same backend RetrievalEvaluator uses internally).

Outputs:
    results/kmer_k{k}/arch_retrieval.json
    results/kmer_k{k}/euk_retrieval.json
    kmer_results_summary.json
"""

import itertools
import json
import os
from collections import defaultdict
from typing import List

import datasets
import numpy as np
import pytrec_eval
from scipy import sparse
from sklearn.preprocessing import normalize

from dgeb.evaluators import RetrievalEvaluator

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

# Use dense encoding + RetrievalEvaluator for k <= this value.
# For k above this, use sparse encoding + direct pytrec_eval.
# k=4 dense costs ~740MB RAM; k=5 dense costs ~118GB — must be sparse.
DENSE_K_MAX = 4

# ─────────────────────────────────────────────────────────────────────────────
# K-MER VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────

def build_kmer_index(k: int) -> dict:
    """Map every standard amino acid k-mer to an integer column index."""
    all_kmers = ["".join(p) for p in itertools.product(STANDARD_AA, repeat=k)]
    return {kmer: idx for idx, kmer in enumerate(all_kmers)}


# ─────────────────────────────────────────────────────────────────────────────
# DENSE ENCODING  (k <= DENSE_K_MAX)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_one_dense(seq: str, kmer_index: dict, k: int) -> np.ndarray:
    """Return a normalized frequency vector (dense float32) for one sequence."""
    vocab_size = len(kmer_index)
    counts = np.zeros(vocab_size, dtype=np.float32)
    seq = seq.upper()
    n_valid = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i: i + k]
        idx = kmer_index.get(kmer)
        if idx is not None:
            counts[idx] += 1
            n_valid += 1
    if n_valid > 0:
        counts /= n_valid
    return counts


def encode_dense(sequences: List[str], k: int) -> np.ndarray:
    """
    Encode sequences as a dense float32 matrix, shape [n_seq, 1, 20^k].
    The middle axis (size 1) is the 'layers' axis expected by RetrievalEvaluator.
    """
    kmer_index = build_kmer_index(k)
    vocab_size = len(kmer_index)
    print(f"  Dense encoding {len(sequences):,} sequences, k={k} "
          f"(vocab={vocab_size:,}, "
          f"~{len(sequences) * vocab_size * 4 / 1e6:.0f} MB)...")
    matrix = np.zeros((len(sequences), vocab_size), dtype=np.float32)
    for i, seq in enumerate(sequences):
        matrix[i] = _encode_one_dense(seq, kmer_index, k)
        if (i + 1) % 2000 == 0:
            print(f"    {i+1:,}/{len(sequences):,}", end="\r")
    print(f"    {len(sequences):,}/{len(sequences):,} done.     ")
    return matrix[:, np.newaxis, :]   # [n_seq, 1, vocab_size]


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE ENCODING  (k >= 5)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_one_sparse(seq: str, kmer_index: dict, k: int, vocab_size: int):
    """
    Return (col_indices, values) for one sequence's sparse row.
    Values are normalized k-mer frequencies.
    """
    counts = {}
    seq = seq.upper()
    n_valid = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i: i + k]
        idx = kmer_index.get(kmer)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1
            n_valid += 1
    if n_valid == 0:
        return [], []
    cols = list(counts.keys())
    vals = [v / n_valid for v in counts.values()]
    return cols, vals


def encode_sparse(sequences: List[str], k: int) -> sparse.csr_matrix:
    """
    Encode sequences as a L2-normalized sparse CSR matrix, shape [n_seq, 20^k].
    Memory cost: O(n_seq * avg_seq_len) — independent of vocab_size.

    Returns a scipy CSR matrix where each row has unit L2 norm,
    ready for cosine similarity via matrix multiplication.
    """
    kmer_index = build_kmer_index(k)
    vocab_size = len(kmer_index)
    n = len(sequences)

    # Build COO data for the CSR matrix
    row_ptrs = [0]
    col_indices = []
    data = []

    print(f"  Sparse encoding {n:,} sequences, k={k} "
          f"(vocab={vocab_size:,}, "
          f"~{n * (350 - k + 1) * 8 / 1e6:.0f} MB estimate)...")

    for i, seq in enumerate(sequences):
        cols, vals = _encode_one_sparse(seq, kmer_index, k, vocab_size)
        col_indices.extend(cols)
        data.extend(vals)
        row_ptrs.append(len(col_indices))
        if (i + 1) % 2000 == 0:
            print(f"    {i+1:,}/{n:,}", end="\r")
    print(f"    {n:,}/{n:,} done.     ")

    # Build CSR
    row_ind = np.array(
        [r for r, (start, end) in enumerate(zip(row_ptrs, row_ptrs[1:]))
         for _ in range(end - start)],
        dtype=np.int32,
    )
    mat = sparse.csr_matrix(
        (np.array(data, dtype=np.float32),
         (row_ind, np.array(col_indices, dtype=np.int32))),
        shape=(n, vocab_size),
    )

    # L2-normalize each row so dot product = cosine similarity
    mat = normalize(mat, norm="l2", copy=False)
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# PYTREC_EVAL WRAPPER  (used by sparse path)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_pytrec(
    sim_matrix: np.ndarray,       # [n_queries, n_corpus]
    query_ids: List[str],
    corpus_ids: List[str],
    qrels: dict,
    k_values: List[int] = [5, 10, 50],
) -> dict:
    """
    Compute retrieval metrics using pytrec_eval directly.
    This is the same backend used inside RetrievalEvaluator.

    Args:
        sim_matrix : cosine similarity scores, shape [n_queries, n_corpus]
        query_ids  : string IDs for each row of sim_matrix
        corpus_ids : string IDs for each column of sim_matrix
        qrels      : {query_id: {corpus_id: relevance}}
        k_values   : cutoffs for @k metrics

    Returns:
        dict of metric_name → mean value across queries (same format as
        RetrievalEvaluator output, so step4 load_map5 works unchanged).
    """
    # Build pytrec_eval run: {query_id: {corpus_id: score}}
    run = {}
    for qi, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        run[qid] = {corpus_ids[ci]: float(sim_matrix[qi, ci])
                    for ci in range(len(corpus_ids))}

    # Measures to compute
    measures = set()
    for k in k_values:
        measures.update([f"map_cut_{k}", f"ndcg_cut_{k}",
                         f"recall_{k}", f"P_{k}"])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_query  = evaluator.evaluate(run)

    # Average across queries and rename to DGEB convention
    # pytrec_eval uses "map_cut_5"; DGEB uses "map_at_5"
    rename = lambda m: m.replace("map_cut_", "map_at_") \
                        .replace("ndcg_cut_", "ndcg_at_") \
                        .replace("recall_", "recall_at_") \
                        .replace("P_", "precision_at_")

    results = {}
    if not per_query:
        return results
    for measure in next(iter(per_query.values())).keys():
        vals = [per_query[qid][measure] for qid in per_query]
        results[rename(measure)] = round(float(np.mean(vals)), 5)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "arch_retrieval": {
        "seq_dataset"  : "tattabio/arch_retrieval",
        "seq_revision" : "a19124322604a21b26b1b3c13a1bd0b8a63c9f7b",
        "qrel_dataset" : "tattabio/arch_retrieval_qrels",
        "qrel_revision": "3f142f2f9a0995d56c6e77188c7251761450afcf",
    },
    "euk_retrieval": {
        "seq_dataset"  : "tattabio/euk_retrieval",
        "seq_revision" : "c93dc56665cedd19fbeaea9ace146f2474c895f0",
        "qrel_dataset" : "tattabio/euk_retrieval_qrels",
        "qrel_revision": "a5aa01e9b9738074aba57fc07434e352c4c71e4b",
    },
}


def load_task_data(task_id: str):
    cfg = TASK_CONFIGS[task_id]
    print(f"  Loading sequences: {cfg['seq_dataset']}")
    seq_ds   = datasets.load_dataset(cfg["seq_dataset"],  revision=cfg["seq_revision"])
    print(f"  Loading qrels    : {cfg['qrel_dataset']}")
    qrels_ds = datasets.load_dataset(cfg["qrel_dataset"], revision=cfg["qrel_revision"])
    corpus_ds = seq_ds["train"]
    query_ds  = seq_ds["test"]
    qrels_dict = defaultdict(dict)
    for row in qrels_ds["train"]:
        qrels_dict[str(row["query_id"])][str(row["corpus_id"])] = int(row["fuzz_ratio"])
    print(f"  Corpus : {len(corpus_ds):,} | Queries: {len(query_ds):,} | "
          f"Qrels: {len(qrels_dict):,} queries")
    return corpus_ds, query_ds, qrels_dict


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL EVALUATION (dispatches dense vs sparse based on k)
# ─────────────────────────────────────────────────────────────────────────────

def run_kmer_retrieval(task_id: str, k: int) -> dict:
    print(f"\nTask: {task_id}, k={k}")
    corpus_ds, query_ds, qrels_dict = load_task_data(task_id)

    if k <= DENSE_K_MAX:
        # ── Dense path: delegate entirely to RetrievalEvaluator ──────────────
        print("Encoding corpus...")
        corpus_embeds = encode_dense(corpus_ds["Sequence"], k)
        print("Encoding queries...")
        query_embeds  = encode_dense(query_ds["Sequence"],  k)

        evaluator = RetrievalEvaluator(
            corpus_embeds=corpus_embeds[:, 0],
            query_embeds =query_embeds[:, 0],
            corpus_ids   =corpus_ds["Entry"],
            query_ids    =query_ds["Entry"],
            qrels        =qrels_dict,
        )
        scores = evaluator()

    else:
        # ── Sparse path: encode → cosine similarity → pytrec_eval ────────────
        print("Encoding corpus (sparse)...")
        corpus_sparse = encode_sparse(corpus_ds["Sequence"], k)  # [n_corpus, vocab]
        print("Encoding queries (sparse)...")
        query_sparse  = encode_sparse(query_ds["Sequence"],  k)  # [n_query,  vocab]

        print("Computing cosine similarity matrix (sparse @ sparse.T)...")
        # Both matrices are L2-normalized so dot product = cosine similarity.
        # Result is [n_queries, n_corpus]; convert to dense for pytrec_eval.
        sim = (query_sparse @ corpus_sparse.T)
        if sparse.issparse(sim):
            sim = sim.toarray()
        sim = np.asarray(sim, dtype=np.float32)
        print(f"  Similarity matrix shape: {sim.shape}, "
              f"~{sim.nbytes / 1e6:.0f} MB")

        print("Evaluating with pytrec_eval...")
        scores = evaluate_with_pytrec(
            sim_matrix =sim,
            query_ids  =list(query_ds["Entry"]),
            corpus_ids =list(corpus_ds["Entry"]),
            qrels      =qrels_dict,
        )

    map_at_5 = scores.get("map_at_5", float("nan"))
    print(f"  MAP@5 = {map_at_5:.5f}")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    k_values  = [1, 2, 3, 4, 5]
    task_ids  = ["arch_retrieval", "euk_retrieval"]
    all_results = {}

    for task_id in task_ids:
        all_results[task_id] = {}
        for k in k_values:
            output_dir  = os.path.join("results", f"kmer_k{k}")
            output_path = os.path.join(output_dir, f"{task_id}.json")
            os.makedirs(output_dir, exist_ok=True)

            # Skip if already cached
            if os.path.exists(output_path):
                print(f"[CACHED] k={k}, {task_id} — skipping.")
                with open(output_path) as f:
                    all_results[task_id][k] = json.load(f)
                continue

            scores = run_kmer_retrieval(task_id, k)
            all_results[task_id][k] = scores
            with open(output_path, "w") as f:
                json.dump(scores, f, indent=2)
            print(f"  Saved to {output_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("K-MER BASELINE RESULTS — MAP@5")
    print("=" * 60)
    print(f"{'Method':<15} {'Arch Retrieval':>16} {'Euk Retrieval':>16}")
    print("-" * 50)
    for k in k_values:
        arch = all_results.get("arch_retrieval", {}).get(k, {}).get("map_at_5", float("nan"))
        euk  = all_results.get("euk_retrieval",  {}).get(k, {}).get("map_at_5", float("nan"))
        tag  = " [sparse]" if k > DENSE_K_MAX else ""
        print(f"{'k=' + str(k):<15} {arch:>16.5f} {euk:>16.5f}{tag}")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "kmer_results": {
            task_id: {str(k): v.get("map_at_5") for k, v in ks.items()}
            for task_id, ks in all_results.items()
        }
    }
    with open("kmer_results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved kmer_results_summary.json")
    print("Next: run step4_plot_results.py")


if __name__ == "__main__":
    main()
