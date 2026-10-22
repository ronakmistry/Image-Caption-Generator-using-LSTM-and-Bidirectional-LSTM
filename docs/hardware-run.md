# Local M5 captioning run

Completed October 21, 2026 (America/New_York) on a MacBook Air with Apple M5,
10-core CPU, 10-core GPU, and 32 GB unified memory.

## Actual experiment

- Frozen, ImageNet-pretrained MobileNet feature encoder (1,024 dimensions).
- Restored original LSTM decoder: three 256-unit recurrent stages, 300-dimensional embedding.
- Official Flickr8k split: 6,000 training, 1,000 validation, 1,000 test images.
- 8,258 vocabulary entries including dedicated padding and unknown tokens; maximum length 40.
- Batch size 512, RMSprop, sparse categorical cross entropy, seed 42, 15 epochs.
- Separate CPython 3.12.14 / TensorFlow 2.18.1 / Keras 3.15.1 / tensorflow-metal 1.2.0 runtime.
- TensorFlow verified `/device:GPU:0` placement on Apple M5 outside the sandbox.

The run first completed 3 epochs, then resumed the saved model and optimizer state for
12 further epochs. No architecture search or test-based checkpoint selection was performed.

## Timings

| Work | Measured time |
|---|---:|
| Feature extraction, all 8,000 split images | 31.61 seconds (253.1 images/second) |
| Training, 15 epochs | 1,832.96 seconds (30.55 minutes) |
| Training plus epoch validation | 1,929.73 seconds (32.16 minutes) |
| Median training epoch | 124.88 seconds |
| Final test caption generation and BLEU calculation, 1,000 images | 4.93 seconds |

These are local interactive-run timings, not an isolated hardware benchmark. Downloads,
installation, warmup, checkpoint writing and preview generation are not all included in
those totals. No old-machine measurements were available, so no speedup factor is claimed.

## Held-out test results

All BLEU scores use whitespace tokens, case preserved, no smoothing, and a 0–1 scale.
These results are specific to **compiled Metal inference**.

| Metric | Value |
|---|---:|
| Mean sentence BLEU-1 | 0.557472 |
| Corpus BLEU-1 | 0.536019 |
| Corpus BLEU-2 | 0.334528 |
| Corpus BLEU-3 | 0.205123 |
| Corpus BLEU-4 | 0.124413 |

The score is not accuracy. The restored preprocessing, dedicated special tokens, and
runtime differ from the coursework, so this does not verify the original saved numbers.
Only MobileNet/LSTM was trained; InceptionV3 and BLSTM reproduction remain pending.

The fixed validation gallery includes both reasonable captions and failures, e.g.
“A black dog is running through the snow .” and an incorrect repeated-shirt description
for the fishing scene. It was selected before training, not chosen for favorable results.

## Inference limitation discovered

**Compiled Metal inference, eager Metal calls, and CPU-only inference do not agree for
this saved checkpoint.** Compiled Metal outputs were repeatable for the fixed validation
examples; the reported test results use that path. The cause is not established, and
cross-device checkpoint portability is not certified. Do not use CPU/eager predictions as
if they reproduce the reported score, and do not assume this is a general Apple/TF issue.
The standard `captioning predict` helper uses eager calls and is not the validated demo
entry point for this particular checkpoint.

Use `scripts/evaluate_hardware.py` from the Metal runtime outside the sandbox. The
per-epoch previews made during training used eager calls and are diagnostic outputs,
not the final compiled inference gallery. The final compiled validation captions are
in `preview-compiled-metal.json`; the final displayed gallery is `caption-preview-metal.png`.

Epochs 1–3 were initially summarized using the last batch because Keras resets its metrics
inside each `*_on_batch` call. Their history entries explicitly identify that limitation.
Epochs 4–15 use weighted epoch means after correcting the reporting. Training updates
were unaffected. The reporting fix has regression tests for partial batches.

## Reproduce or inspect locally

From the repository root, with a separate virtual environment at `../.venv-caption-run`:

```bash
python3.12 -m venv ../.venv-caption-run
../.venv-caption-run/bin/python -m pip install -r requirements-metal.txt
../.venv-caption-run/bin/python scripts/prepare_flickr8k.py
../.venv-caption-run/bin/python scripts/run_hardware.py --device gpu --epochs 15 --batch-size 512
../.venv-caption-run/bin/python scripts/evaluate_hardware.py
../.venv-caption-run/bin/python scripts/evaluate_hardware.py --validation-preview
```

The preparation script downloads the Flickr8k mirror from the `jbrownlee/Datasets` GitHub
release and official Keras MobileNet no-top weights from Google Storage. Source URLs,
byte counts and SHA-256 values are stored in `data/flickr8k/provenance.json`. Review Flickr8k
usage terms before redistributing its images. The model, dataset, and gallery remain local
and ignored by Git.

Existing local artifacts are in `artifacts/hardware-run/`: `model.keras` (51.3 MiB including
optimizer state), `metadata.json`, `history.json`, `benchmark-gpu.json`, `complete.json`,
`test-results.json`, `preview-compiled-metal.json`, `caption-preview-metal.png`, and
`training-loss.png`. Cached pretrained image features are under `data/flickr8k/`.
