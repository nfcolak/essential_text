# Split assignments

Text-free split assignments derived from the locked manifests in `data/splits/` (checksums in `SHA256SUMS`).

## Files

- `<config>.csv` (one per Financial PhraseBank agreement config): columns
  `id,sha256,label,cluster_id,split,cv_fold`. `id` is the sentence id, `sha256` the hash of the normalised
  sentence, `label` is 0/1/2 = negative/neutral/positive, `cluster_id` the near-duplicate cluster (the whole
  cluster sits in one split), `split` is train/val/test, `cv_fold` is the 5-fold CV fold whose validation part
  holds the sentence (train ids only; empty otherwise). Rows are sorted by id.
- `sentfin_holdout.csv`: columns `id,sha256,label` of the deduplicated SEntFiN hold-out sentences.

No sentence text is stored here. Texts come from the pinned Financial PhraseBank zip and the pinned SEntFiN csv
(see `src/data/download.py` for URLs and SHA-256 sums); match them by `sha256`.

## Rebuild and check

Rebuild the manifests: `python3 colab/run_step1_colab.py ... --fpb-config all`. Then verify these files:

    python3 scripts/export_split_assignments.py --check

The check first verifies the inputs against `SHA256SUMS` (exit 2 on mismatch), then compares the rebuilt rows with
the files here (exit 0 identical, 1 otherwise).
