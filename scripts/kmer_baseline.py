"""
K-mer frequency baseline for DGEB retrieval tasks.

NOTE on qrels structure (discovered in step1):
  The Arch Retrieval qrels have 163,612 rows across 2,343 queries,
  meaning ~69.83 relevant corpus entries per query on average.
  Relevance label is 1 (fuzz_ratio > 90 during dataset construction),
  stored as fuzz_ratio=1 in the HuggingFace dataset.
  pytrec_eval treats any score > 0 as relevant, so MAP@5 measures
  whether any of the ~70 relevant bacterial proteins appear in the
  top-5 retrieved results. This is a retrieval-with-many-positives
  regime, not a needle-in-a-haystack regime.

We implement a KmerModel whose encode() method returns k-mer frequency vectors
with exactly the same shape contract as BioSeqTransformer.encode():
    output shape: [n_sequences, n_layers, embed_dim]
        n_layers = 1   (k-mer has no layer concept; one representation per k)
        embed_dim = 20^k  (number of possible k-mers over standard amino acids)

We feed these vectors directly into DGEB's RetrievalEvaluator, which
computes MAP@5 identically to the foundation model evaluation.

We evaluate k in {1, 2, 3, 4} on both Arch and Euk retrieval tasks.

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
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from dgeb.evaluators import RetrievalEvaluator

# Standard 20 amino acids. Non-standard characters (B, Z, X, U, O) are ignored.
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

# ─────────────────────────────────────────────────────────────────────────────
# K-MER ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def build_kmer_index(k: int) -> dict:
    """
    Build a mapping from every k-mer string to an integer index.

    The vocabulary is all 20^k combinations of standard amino acids,
    in lexicographic order. Non-standard characters are ignored during
    encoding (the k-mer containing them is simply skipped).

    Returns:
        dict: {kmer_string -> int_index}, length = 20^k
    """
    all_kmers = ["".join(p) for p in itertools.product(STANDARD_AA, repeat=k)]
    return {kmer: idx for idx, kmer in enumerate(all_kmers)}


def encode_sequence(seq: str, kmer_index: dict, k: int) -> np.ndarray:
    """
    Encode a single amino acid sequence as a normalized k-mer frequency vector.

    Algorithm:
        1. Slide a window of width k across the sequence.
        2. Count each k-mer that contains only standard amino acids.
        3. Divide counts by total number of valid k-mers (L2-normalize
           is NOT applied here; we use L2 normalization across all
           sequences together after stacking, because sklearn's
           cosine_similarity handles un-normalized vectors correctly,
           but the RetrievalEvaluator uses cos_sim which normalizes
           internally via F.normalize — so raw counts are fine).

    Args:
        seq:        Amino acid string (arbitrary case).
        kmer_index: Output of build_kmer_index(k).
        k:          K-mer length.

    Returns:
        np.ndarray of shape (20^k,), dtype float32.
    """
    vocab_size = len(kmer_index)
    counts = np.zeros(vocab_size, dtype=np.float32)

    seq = seq.upper()
    n_valid = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        idx  = kmer_index.get(kmer)   # None if any non-standard character
        if idx is not None:
            counts[idx] += 1
            n_valid += 1

    if n_valid > 0:
        counts /= n_valid   # normalize to frequencies (sum to 1)
    # If n_valid == 0 (sequence has no valid k-mers), return zero vector.
    # This should not happen for standard protein sequences.
    return counts


def encode_sequences(sequences: List[str], k: int) -> np.ndarray:
    """
    Encode a list of sequences and return an array compatible with
    BioSeqTransformer.encode() output shape.

    Args:
        sequences: List of amino acid strings.
        k:         K-mer length.

    Returns:
        np.ndarray of shape [n_sequences, 1, 20^k]
            - axis 0: sequences
            - axis 1: "layers" (always 1 for k-mer model)
            - axis 2: embedding dimensions (20^k k-mer frequencies)
    """
    kmer_index = build_kmer_index(k)
    vocab_size = len(kmer_index)   # 20^k

    print(f"  Encoding {len(sequences):,} sequences with k={k} "
          f"(vocab size = {vocab_size:,})...")
    matrix = np.zeros((len(sequences), vocab_size), dtype=np.float32)
    for i, seq in enumerate(sequences):
        matrix[i] = encode_sequence(seq, kmer_index, k)
        if (i + 1) % 1000 == 0:
            print(f"    {i+1:,}/{len(sequences):,}", end="\r")
    print(f"    {len(sequences):,}/{len(sequences):,} done.     ")

    # Shape: [n_sequences, 1, vocab_size]
    # axis 1 = "layers" — DGEB evaluator indexes corpus_embeds[:, i]
    return matrix[:, np.newaxis, :]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (mirrors retrieval_tasks.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "arch_retrieval": {
        "display_name" : "Arch Retrieval",
        "seq_dataset"  : "tattabio/arch_retrieval",
        "seq_revision" : "a19124322604a21b26b1b3c13a1bd0b8a63c9f7b",
        "qrel_dataset" : "tattabio/arch_retrieval_qrels",
        "qrel_revision": "3f142f2f9a0995d56c6e77188c7251761450afcf",
    },
    "euk_retrieval": {
        "display_name" : "Euk Retrieval",
        "seq_dataset"  : "tattabio/euk_retrieval",
        "seq_revision" : "c93dc56665cedd19fbeaea9ace146f2474c895f0",
        "qrel_dataset" : "tattabio/euk_retrieval_qrels",
        "qrel_revision": "a5aa01e9b9738074aba57fc07434e352c4c71e4b",
    },
}


def load_task_data(task_id: str):
    """
    Load corpus, queries, and qrels for a retrieval task.
    Returns corpus_ds, query_ds, qrels_dict — same format used by
    DGEB's run_retrieval_task().
    """
    cfg = TASK_CONFIGS[task_id]
    print(f"  Loading sequences: {cfg['seq_dataset']}")
    seq_ds   = datasets.load_dataset(cfg["seq_dataset"],  revision=cfg["seq_revision"])
    print(f"  Loading qrels    : {cfg['qrel_dataset']}")
    qrels_ds = datasets.load_dataset(cfg["qrel_dataset"], revision=cfg["qrel_revision"])

    # retrieval_tasks.py uses train=corpus, test=queries
    corpus_ds = seq_ds["train"]
    query_ds  = seq_ds["test"]

    # Build qrels dict: {query_id: {corpus_id: relevance_score}}
    # Relevance score = fuzz_ratio (>90 means relevant per DGEB Appendix A)
    qrels_dict = defaultdict(dict)
    for row in qrels_ds["train"]:
        qrels_dict[str(row["query_id"])][str(row["corpus_id"])] = int(row["fuzz_ratio"])

    print(f"  Corpus : {len(corpus_ds):,} sequences")
    print(f"  Queries: {len(query_ds):,} sequences")
    print(f"  Qrels  : {len(qrels_dict):,} unique queries with relevance labels")
    return corpus_ds, query_ds, qrels_dict


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_kmer_retrieval(task_id: str, k: int) -> dict:
    """
    Run k-mer retrieval on one task and return the metrics dict.

    This mirrors the logic of DGEB's run_retrieval_task() but replaces
    model.encode() with our k-mer encode_sequences().

    The RetrievalEvaluator expects:
        corpus_embeds : np.ndarray of shape [n_corpus, embed_dim]
        query_embeds  : np.ndarray of shape [n_queries, embed_dim]
        corpus_ids    : list of string IDs
        query_ids     : list of string IDs
        qrels         : {query_id: {corpus_id: relevance_score}}

    It computes cosine similarity internally (eval_utils.cos_sim uses
    F.normalize then matrix multiplication), so our un-normalized
    frequency vectors are handled correctly.
    """
    print(f"\nTask: {task_id}, k={k}")
    corpus_ds, query_ds, qrels_dict = load_task_data(task_id)

    # Encode: output shape [n_sequences, 1, 20^k]
    print("Encoding corpus...")
    corpus_embeds = encode_sequences(corpus_ds["Sequence"], k)  # [n_corpus, 1, vocab]
    print("Encoding queries...")
    query_embeds  = encode_sequences(query_ds["Sequence"],  k)  # [n_query, 1, vocab]

    # Layer index 0 = our single "layer"
    # corpus_embeds[:, 0] has shape [n_corpus, vocab_size]
    layer_idx = 0
    evaluator = RetrievalEvaluator(
        corpus_embeds=corpus_embeds[:, layer_idx],   # [n_corpus, vocab_size]
        query_embeds =query_embeds[:, layer_idx],    # [n_query,  vocab_size]
        corpus_ids   =corpus_ds["Entry"],
        query_ids    =query_ds["Entry"],
        qrels        =qrels_dict,
    )
    scores = evaluator()

    # Primary metric is map_at_5
    map_at_5 = scores.get("map_at_5", float("nan"))
    print(f"  MAP@5 = {map_at_5:.5f}")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: run all (k, task) combinations and collect results
# ─────────────────────────────────────────────────────────────────────────────

def main():
    k_values    = [1, 2, 3, 4]
    task_ids    = ["arch_retrieval", "euk_retrieval"]
    all_results = {}   # {task_id: {k: scores_dict}}

    for task_id in task_ids:
        all_results[task_id] = {}
        for k in k_values:
            output_dir  = os.path.join("results", f"kmer_k{k}")
            output_path = os.path.join(output_dir, f"{task_id}.json")
            os.makedirs(output_dir, exist_ok=True)

            scores = run_kmer_retrieval(task_id, k)
            all_results[task_id][k] = scores

            with open(output_path, "w") as f:
                json.dump(scores, f, indent=2)
            print(f"  Saved to {output_path}")

    # ── Print final comparison table ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("K-MER BASELINE RESULTS — MAP@5")
    print("=" * 70)
    print(f"{'Method':<20} {'Arch Retrieval':>18} {'Euk Retrieval':>18}")
    print("-" * 58)

    for k in k_values:
        arch = all_results["arch_retrieval"][k].get("map_at_5", float("nan"))
        euk  = all_results["euk_retrieval"][k].get("map_at_5",  float("nan"))
        print(f"{'k-mer k=' + str(k):<20} {arch:>18.5f} {euk:>18.5f}")

    # Save summary
    summary = {
        "kmer_results": {
            task_id: {str(k): v.get("map_at_5") for k, v in ks.items()}
            for task_id, ks in all_results.items()
        }
    }
    with open("kmer_results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to kmer_results_summary.json")


if __name__ == "__main__":
    main()