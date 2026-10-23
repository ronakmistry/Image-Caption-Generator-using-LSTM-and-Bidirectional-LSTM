# A bidirectional decoder alternative

Completed locally in October 2026 on the Apple M5 MacBook Air with 32 GB memory.

## Result

A smaller bidirectional decoder changes several captions usefully, but does **not** beat
the saved LSTM overall. Keep the LSTM as the default; retain this BLSTM for comparison.

| Full validation set: 1,000 images | Existing LSTM | New BLSTM |
|---|---:|---:|
| Corpus BLEU-1, beam width 3 | 0.632902 | 0.610331 |
| Corpus BLEU-4, beam width 3 | **0.185272** | 0.166946 |
| Adjacent repeated-token rate | 0.000942 | 0.000000 |

The BLSTM's greedy BLEU-4 is 0.150135; its beam-search BLEU-4 is 0.166946. Neither
metric is a factual-accuracy percentage. The test split was **not evaluated** during
this experiment and was not used for model selection.

## What changed

- Kept the original three recurrent stages and image/text merge. Used **64 units per
  direction**, compared with 256 units in the saved unidirectional decoder, to explore
  a smaller model within a practical local compute budget. BLSTM parameter count: 3,841,202.
- Trained the decoder from random initialization on all 6,000 official training images.
  Reused the same frozen ImageNet MobileNet features, training vocabulary and preprocessing.
  No additional data or weights were downloaded.
- Used RMSprop at 0.001, batch size 512, seed 42. Configured learning-rate halving after
  two epochs without a 0.005 validation-loss improvement and stopping after four such epochs.
  Those controls did not trigger before the runtime stop; the learning rate stayed at 0.001.
- Ran **13 completed epochs**, retaining epoch 13 by minimum validation loss (3.174030).
  The planned maximum was 15; the run was stopped for runtime budget, not convergence.
  Completed epoch timings sum to **4,067.47 seconds (67.79 minutes)**. Final validation
  decoding added roughly 28 seconds. These recorded intervals exclude the earlier,
  interrupted 256-unit attempt and do not represent overnight wall-clock duration.
- Compared fixed greedy and beam-width-3 decoding (length penalty 0.6) on validation.
  This is an exploratory comparison: model size, recurrence direction and training budget
  differ from the saved epoch-23 LSTM. It does not isolate the effect of bidirectionality.

## Runtime checks

The original 256-unit-per-direction attempt was interrupted before a completed epoch
after an overnight interval. That alone does not establish a hardware or TensorFlow hang.
Full-size benchmarks then measured approximately 1.60 seconds per Metal batch, versus
0.22–0.23 seconds per CPU batch for the smaller model. The measured CPU path was used.
Synthetic batch timings were estimates; full real-data epochs took longer.

`PortableBidirectional` uses forward LSTM kernels on the original and valid reversed
prefixes. It reverses backward sequence outputs back into alignment, concatenates both
directions and zeros masked sequence positions, matching the original masked BLSTM.
No future target tokens enter either direction.

- Synthetic CPU tests compare native and portable BLSTM probabilities, gradients,
  training loss and save/load behavior.
- A CPU/Metal probe covers prefix lengths 1, 2, 19 and 40. The maximum probability
  difference was 5.59e-9; parameter gradient differences were at most 1.05e-7.
- The trained checkpoint was independently reloaded on both runtimes. On 12 validation
  images, maximum probability difference was **1.34e-6** and all beam captions matched.
- The existing LSTM checkpoint's SHA-256 was unchanged.

## What the images show

The gallery contains all six previously shown validation images, followed by six extra
images sampled without replacement with seed 20261021. Examples were fixed before the
BLSTM comparison and were not substituted based on favorable outputs.

- Two dogs: LSTM says one dog with a red ball; BLSTM says **two dogs in a grassy field**.
  The BLSTM gets the count but omits the ball.
- Surfer: LSTM says a young boy; BLSTM says **a man in a wetsuit surfing**.
- Dock: BLSTM recognizes jumping but invents a red shirt and says "over" a lake.
- Soccer: both invent a red uniform and miss the two-player interaction.
- Fishing children and a child on a trampoline: both models miss important subjects/actions.
- Additional images: BLSTM recognizes snowy terrain and a skateboard where the LSTM
  predicts a cliff and a bicycle. Other details remain incorrect.

This small visual inspection is descriptive, not a statistically measured improvement
in factual accuracy. More training is not guaranteed to solve these grounding errors.

## Local artifacts and commands

Artifacts remain ignored under `artifacts/blstm-comparison/`: `model.keras`,
`metadata.json`, `protocol.json`, `history.json`, `complete.json`, both validation
prediction files, `preview.json`, consistency probes and `comparison-1.png` / `comparison-2.png`.
The earlier LSTM remains under `artifacts/improvement-run/`.

Use the alternative with the normal prediction helper:

```bash
../.venv/bin/python -m captioning predict \
  --artifact-dir artifacts/blstm-comparison \
  --image data/flickr8k/Flicker8k_Dataset/3525403875_7f95e0cbfe.jpg \
  --encoder-weights data/flickr8k/downloads/mobilenet_1_0_224_tf_no_top.h5
```

To run a new CPU comparison without overwriting the saved experiment:

```bash
../.venv/bin/python scripts/compare_blstm.py --probe cpu --output artifacts/blstm-repeat
../.venv/bin/python scripts/compare_blstm.py --train --train-device cpu --units 64 \
  --epochs 13 --time-budget-seconds 0 --output artifacts/blstm-repeat
../.venv/bin/python scripts/compare_blstm.py --verify cpu --output artifacts/blstm-repeat
```

The normal training default caps the run at 15 epochs or an elapsed 50-minute budget,
checked after each completed epoch. The command above explicitly requests 13 epochs
without the time limit. A completed experiment is protected against overwrite.
Metal probes require the optional `tensorflow-metal==1.2.0` runtime and GPU access;
the pinned base environment supports CPU execution.

The existing restoration PR contains the portable BLSTM implementation, synthetic tests,
comparison runner and gallery renderer. Model files and dataset photos are kept local.
