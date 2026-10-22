# Captioning improvement experiment

Completed October 21, 2026 on the same Apple M5 MacBook Air (32 GB).

## Result

Ten additional epochs were run (16–25); validation selected checkpoint **23**.
The learning rate fell from 0.001 to 0.0005 after epoch 22. A fixed comparison on all
1,000 validation images selected beam search, width 3, length penalty 0.6.

| Validation metric, same selected checkpoint | Greedy | Beam width 3 |
|---|---:|---:|
| Corpus BLEU-1 | 0.604544 | 0.632902 |
| Corpus BLEU-4 | 0.170051 | 0.185272 |
| Adjacent repeated-token rate | 0.002455 | 0.000942 |

After model and decoding selection, one final evaluation over the 1,000 test images gave:

| Test metric | Value |
|---|---:|
| Mean sentence BLEU-1 | 0.627294 |
| Corpus BLEU-1 | 0.649918 |
| Corpus BLEU-2 | 0.431131 |
| Corpus BLEU-3 | 0.290654 |
| Corpus BLEU-4 | 0.196225 |

Total experiment time, including baseline validation, training, decoding comparison and
final evaluation: **1,401.43 seconds (23.36 minutes)**. Early stopping was configured with
patience 4 but did not trigger; the ten-epoch budget was reached. The best checkpoint
was restored instead of using the final epoch. Validation loss at epoch 23 was 3.207807;
epochs 24 and 25 worsened to 3.225576 and 3.234961 while training loss kept decreasing.
This suggests caution about simply adding more epochs at the same settings.

## Correctness fix before continuation

The previous checkpoint produced inconsistent predictions across CPU, eager Metal and
compiled Metal. Identical saved weight hashes ruled out a load/version mismatch.
Layer probes identified two execution differences:

- Metal's accelerated recurrent path differed in padded sequence/final-state outputs.
  `GatherLSTM` takes the last valid sequence output explicitly and normalizes masked
  positions, rather than using the inconsistent returned final-state value.
- Dense/ReLU graph fusion differed from CPU by large amounts. `PortableDense` expresses
  bias addition as `AddV2` to avoid that fused path. A matching independently reported
  case is documented on the [Apple developer forum](https://developer.apple.com/forums/thread/827708).
  That report corroborates the Dense issue; it does not establish the recurrent issue's
  internal vendor cause.

These changes preserve the intended unidirectional LSTM equations and learned weight
shapes. On controlled variable-length inputs, corrected forward probabilities and all
18 parameter gradients agreed across CPU/Metal (gradient maximum absolute difference
1.37e-6). The final saved checkpoint was reloaded independently on each runtime:

- CPU vs Metal maximum probability difference on the fixed probe: **5.99e-6**.
- Eager vs compiled probability difference: **0** on both runtimes for that probe.
- Beam captions on the same six validation images: **identical across CPU and Metal**.

This is targeted validation for the current decoder and environment, not a claim about
all TensorFlow models or all Metal versions. The wrapper is for right-padded,
unidirectional, nonempty caption prefixes; BLSTM acceleration is not validated here.

The old 15-epoch weights were preserved and used as the starting point, with optimizer
state copied without altering parameter ordering. Corrected execution initially exposed
poor captions from those weights. Therefore the earlier 0.124 test BLEU-4 from the
inconsistent execution path is **not an apples-to-apples improvement baseline**.
The new gain cannot be attributed to more epochs alone: it combines backend corrections,
continued training and decoding selection. The clean greedy-vs-beam comparison above
isolates the decoding effect on the same corrected checkpoint.

## Controls and remaining errors

- Original 15-epoch model unchanged; verified by SHA-256.
- Original official Flickr8k train/validation/test image split retained.
- Checkpoint selected only by validation loss; decoding selected only by validation BLEU-4.
- Learning-rate reduction: factor 0.5 after two epochs without >=0.005 validation-loss
  improvement, floor 1e-5. Early stopping after four such epochs.
- Same six-image gallery chosen before the original training. No favorable examples
  substituted. Captions still hallucinate colors, counts, ages and actions; e.g. a yellow
  ball is called red, and a person jumping is described as standing.
- The final test set was read only after selection; test scores did not drive choices.

## Use the improved model locally

From this repository root, using the existing CPU environment:

```bash
../.venv/bin/python -m captioning predict \
  --artifact-dir artifacts/improvement-run \
  --image data/flickr8k/Flicker8k_Dataset/3375920709_ef7370fa09.jpg \
  --encoder-weights data/flickr8k/downloads/mobilenet_1_0_224_tf_no_top.h5
```

The same command works in `../.venv-caption-run` with Metal access. The prediction helper
imports the custom checkpoint layers and honors the saved beam-search configuration.
The selected checkpoint is `artifacts/improvement-run/model.keras`; the untouched starting
point remains in `artifacts/hardware-run/model.keras`. Data, models and galleries remain
local and ignored by Git.

The experiment is reproducible via `scripts/improve_captioning.py` (it refuses to overwrite
a completed experiment). `scripts/verify_improved.py --device cpu` and `--device metal`
check the saved probabilities and captions. Local evidence is in `complete.json`,
`history.json`, `selection.json`, `validation-greedy.json`, `validation-beam3.json`,
`test-results.json`, `consistency-summary.json`, and `caption-preview.png` under the
improvement artifact directory. Small synthetic tests cover mask propagation,
serialization, gradients, beam selection and BLEU/repetition calculations.
