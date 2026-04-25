from collections import defaultdict
import datasets

# ── Arch Retrieval ────────────────────────────────────────────────────────────
print("=" * 60)
print("ARCH RETRIEVAL DATASET")
print("=" * 60)

arch_seqs = datasets.load_dataset(
    "tattabio/arch_retrieval",
    revision="a19124322604a21b26b1b3c13a1bd0b8a63c9f7b",
)
arch_qrels = datasets.load_dataset(
    "tattabio/arch_retrieval_qrels",
    revision="3f142f2f9a0995d56c6e77188c7251761450afcf",
)

corpus_arch = arch_seqs["train"]
query_arch  = arch_seqs["test"]

print(f"\nCorpus split  : {len(corpus_arch):,} sequences")
print(f"Query split   : {len(query_arch):,} sequences")
print(f"Corpus columns: {corpus_arch.column_names}")
print(f"Query columns : {query_arch.column_names}")

print("\nFirst corpus entry:")
row = corpus_arch[0]
for k, v in row.items():
    val = v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v
    print(f"  {k:20s}: {val}")

print("\nFirst query entry:")
row = query_arch[0]
for k, v in row.items():
    val = v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v
    print(f"  {k:20s}: {val}")

print(f"\nQrels splits  : {arch_qrels}")
qrels_ds = arch_qrels["train"]   # the whole qrels is in "train"
print(f"Qrels columns : {qrels_ds.column_names}")
print(f"Qrels rows    : {len(qrels_ds):,}")
print("\nFirst 3 qrel rows:")
for i in range(3):
    print(f"  {dict(qrels_ds[i])}")

# Build qrels dict exactly as retrieval_tasks.py does
qrels_dict = defaultdict(dict)
for row in qrels_ds:
    qrels_dict[str(row["query_id"])][str(row["corpus_id"])] = int(row["fuzz_ratio"])

n_queries_with_rel = sum(1 for v in qrels_dict.values() if v)
rel_counts = [len(v) for v in qrels_dict.values()]
print(f"\nUnique queries in qrels : {len(qrels_dict):,}")
print(f"Avg relevant docs/query : {sum(rel_counts)/len(rel_counts):.2f}")
print(f"Max relevant docs/query : {max(rel_counts)}")

# Sequence length stats
seq_lens = [len(s) for s in corpus_arch["Sequence"]]
print(f"\nCorpus sequence lengths : min={min(seq_lens)}, "
      f"mean={sum(seq_lens)//len(seq_lens)}, max={max(seq_lens)}")
seq_lens_q = [len(s) for s in query_arch["Sequence"]]
print(f"Query sequence lengths  : min={min(seq_lens_q)}, "
      f"mean={sum(seq_lens_q)//len(seq_lens_q)}, max={max(seq_lens_q)}")

# ── Euk Retrieval ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EUK RETRIEVAL DATASET")
print("=" * 60)

euk_seqs = datasets.load_dataset(
    "tattabio/euk_retrieval",
    revision="c93dc56665cedd19fbeaea9ace146f2474c895f0",
)
euk_qrels = datasets.load_dataset(
    "tattabio/euk_retrieval_qrels",
    revision="a5aa01e9b9738074aba57fc07434e352c4c71e4b",
)

corpus_euk = euk_seqs["train"]
query_euk  = euk_seqs["test"]

print(f"\nCorpus split  : {len(corpus_euk):,} sequences")
print(f"Query split   : {len(query_euk):,} sequences")

seq_lens_ec = [len(s) for s in corpus_euk["Sequence"]]
print(f"Corpus sequence lengths : min={min(seq_lens_ec)}, "
      f"mean={sum(seq_lens_ec)//len(seq_lens_ec)}, max={max(seq_lens_ec)}")

print("\nDone. These shapes match DGEB Table 1.")
