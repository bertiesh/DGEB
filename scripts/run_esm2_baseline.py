"""
Run DGEB retrieval tasks with ESM2-8M (smallest foundation model).

Expected results from DGEB Appendix F (esm2_t6_8M):
    Arch Retrieval MAP@5 : 0.179
    Euk  Retrieval MAP@5 : 0.215

Outputs results JSON to: results/esm2_t6_8M_UR50D/
"""

import dgeb


def run():
    MODEL_NAME    = "facebook/esm2_t6_8M_UR50D"
    OUTPUT_FOLDER = "results"

    print(f"Loading model: {MODEL_NAME}")
    model = dgeb.get_model(
        model_name=MODEL_NAME,
        layers=None,           # evaluate both mid and last layer (DGEB default)
        devices=[0],           # uses CPU automatically if CUDA unavailable
        max_seq_length=1024,
        batch_size=32,
        pool_type="mean",
        num_processes=0,       # disables DataLoader multiprocessing
    )
    print(f"  Modality   : {model.modality}")
    print(f"  Num layers : {model.num_layers}")
    print(f"  Embed dim  : {model.embed_dim}")
    print(f"  Layers eval: {model.layers}")

    # Filter to retrieval tasks only
    all_protein_tasks = dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN)
    retrieval_tasks   = [t for t in all_protein_tasks
                         if t.metadata.type == "retrieval"]

    print(f"\nRunning {len(retrieval_tasks)} retrieval tasks:")
    for t in retrieval_tasks:
        print(f"  - {t.metadata.display_name}")

    evaluation = dgeb.DGEB(tasks=retrieval_tasks)
    results    = evaluation.run(model, output_folder=OUTPUT_FOLDER)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for task_result in results:
        task_name         = task_result.task.display_name
        primary_metric_id = task_result.task.primary_metric_id

        print(f"\n{task_name}")
        for layer_result in task_result.results:
            for metric in layer_result.metrics:
                if metric.id == primary_metric_id:
                    print(f"  Layer {layer_result.layer_display_name:>6}: {metric.value:.5f}")

        best = max(
            metric.value
            for lr in task_result.results
            for metric in lr.metrics
            if metric.id == primary_metric_id
        )
        print(f"  Best MAP@5    : {best:.5f}")

    print(f"\nJSON results saved to: {OUTPUT_FOLDER}/{MODEL_NAME.split('/')[-1]}/")
    

if __name__ == "__main__":
    run()