# Image caption generation with LSTM and bidirectional LSTM

**Latest local result:** corrected CPU/Metal execution, ten further training epochs, and validation-selected beam search achieved test corpus BLEU-4 **0.1962**. See the [improvement experiment](docs/improvement-run.md) for evidence, commands and comparison limits.

A restored 2018 graduate project by **Abhinav Tiwari, Ronak Mistry, and Shivam Negi**.
The experiment asks whether bidirectional recurrence improves caption generation enough
to justify its additional computation, across InceptionV3 and MobileNet image encoders.

## Method

Frozen ImageNet encoders produce 2,048-dimensional InceptionV3 or 1,024-dimensional
MobileNet image features. Each decoder retains the original three recurrent stages,
image/text merge, dropout, and next-token softmax prediction. Default recurrent width is
256 with a 300-dimensional embedding/projection. Both LSTM and BLSTM variants are supported.
Sparse integer targets replace memory-heavy one-hot targets without changing the objective.

The restored pipeline uses the official Flickr8k train/dev/test image lists. Vocabulary
and maximum sequence length come only from training captions. Padding is always token 0;
unknown words have their own token. Recurrent masks exclude padding. Caption generation
uses greedy decoding, suppresses padding/start outputs, and stops at end-of-caption or
the configured length limit. BLSTM processes the available prefix, never future target words.

Compared with the historical source, the restoration fixes wrong-model evaluation,
unordered vocabulary indices, a padding/word index collision, the MobileNet generator's
incorrect training filename, discarded partial batches, and hardcoded evaluation counts.
Supported modules replace deprecated Keras imports and `fit_generator` calls.

## Setup and verification

Use Python **3.12** in an isolated environment. CPU execution is the default.
Run these commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
python scripts/execute_notebooks.py
```

On Windows, activate with `.venv\Scripts\activate`. The notebook runner starts a fresh
kernel for each supported root notebook and writes executed copies and HTML to
`artifacts/notebooks/`. Open the HTML locally to read the rendered walkthrough.
Root notebooks default to generated fixtures and perform no downloads.
Direct dependencies are pinned in `requirements.txt`; `requirements-lock.txt`
records the complete tested environment. The lock targets CPython 3.12 on macOS
Apple Silicon and Linux; optional platform packages use environment markers.
CI installs the lock, runs fixture tests, executes notebooks, and uploads the rendered results.

## Evidence and preservation

**The restoration checks below use fixtures. A subsequent 15-epoch MobileNet/LSTM run on real Flickr8k data is documented in [the M5 run report](docs/hardware-run.md), including an inference portability limitation.** Fixture metrics
validate code paths only. They do not establish model quality or recover historical results.
See [verification](docs/verification.md) for checks and limitations, and
[inventory](docs/inventory.json) for the starting commit, original files, and missing resources.
The original notebook sources and saved outputs are preserved in [historical/](historical/).
They use obsolete APIs and are not supported execution entry points.

Generated data, features, models, reports, and notebook outputs stay under ignored
`artifacts/` or `data/` directories. Review dataset permissions before adding any data to Git.

## Quick run: generated fixtures

```bash
python -m captioning smoke --encoder mobilenet --variant lstm --output artifacts/mobile-lstm-smoke
python -m captioning smoke --encoder mobilenet --variant blstm --output artifacts/mobile-blstm-smoke
python -m captioning smoke --encoder inception --variant lstm --output artifacts/inception-lstm-smoke
python -m captioning smoke --encoder inception --variant blstm --output artifacts/inception-blstm-smoke
```

Smoke mode generates four small random RGB images, runs the real feature-extraction
architecture with **random encoder weights**, uses a reduced decoder (8 recurrent units,
12 embedding dimensions), trains one epoch, evaluates a disjoint fixture image, and
verifies a save/load prediction round trip. It has no pretrained-language or image knowledge.

## Real data and training: pending resource recovery

Supply this layout manually; the code never downloads data or weights:

```text
data/flickr8k/
  Flicker8k_Dataset/                 # original spelling: Flicker
    <image-id>.jpg
  Flickr8k_text/
    Flickr8k.token.txt
    Flickr_8k.trainImages.txt
    Flickr_8k.devImages.txt
    Flickr_8k.testImages.txt
```

Also supply compatible **ImageNet no-top weights** for the chosen encoder in a local file.
The legacy PDFs describe the original data acquisition and Python 3.6 environment; their
CUDA 9 / TensorFlow GPU instructions are historical and should not be used for this setup.
Obtain the original Flickr8k release and check its terms before redistribution. Missing
files produce actionable errors rather than silently substituting another dataset.

```bash
python -m captioning train --data-root data/flickr8k --encoder mobilenet --variant blstm --encoder-weights data/mobilenet.weights.h5 --epochs 15 --batch-size 512 --output artifacts/mobile-blstm
python -m captioning predict --artifact-dir artifacts/mobile-blstm --image data/example.jpg --encoder-weights data/mobilenet.weights.h5
```

Repeat training for the other three encoder/decoder combinations with the matching weight
file and a separate output directory. Full runs may be slow on CPU. No paid compute or
full benchmark training is part of this pass. The supported runtime intentionally selects
CPU; GPU execution would require an explicitly validated runtime change.

Each run writes `model.keras`, `metadata.json` (vocabulary, splits, preprocessing and
configuration), `features.npz`, and `results.json` (metrics, predictions, training history,
seed, environment and resource fingerprints). Feature caches are regenerated by each run;
they are an inspectable output, not an independently versioned input. Use the saved metadata
with the saved model; historical weights cannot be mixed with a newly generated vocabulary.
External-image prediction verifies that encoder weights match the training fingerprint.

## Evaluation and historical results

The original conclusions report BLEU values around **0.57–0.59**. These are **historical,
unverified**, and the wrong-model selection bug means the original MobileNet comparison
cannot substantiate the BLSTM advantage. Do not interpret them as accuracy percentages.

The closest restored historical metric is **mean sentence BLEU-1**: compute NLTK sentence
BLEU with unigram weights `(1, 0, 0, 0)` against all reference captions, then average across
the actual evaluated image count. This incorporates the brevity penalty and multi-reference
matching; it does not select a single best reference score. New reports also include
**corpus BLEU-1 through BLEU-4**, with uniform n-gram weights, whitespace tokenization,
case preserved, and no smoothing. Scores are on a 0–1 scale. Empty generated captions
score zero. Different aggregation/tokenization choices are not directly comparable.

BLEU alone does not measure caption usefulness. A future real-data run should inspect
representative successes and failures and report runtime alongside model quality.

## Notebook guide and attribution

- `Image_Captioning.ipynb`: InceptionV3 LSTM/BLSTM walkthrough.
- `Mobilenet_ImageCap.ipynb`: MobileNet LSTM/BLSTM walkthrough.
- `Readme.pdf` and `Tensorflow GPU Setup.pdf`: preserved original project notes.

Original collaborative attribution and the existing [MIT license](LICENSE) are retained.
References: [Keras](https://keras.io/), [NLTK BLEU implementation](https://www.nltk.org/api/nltk.translate.bleu_score.html),
[original Flickr8k research](https://www.jair.org/index.php/jair/article/view/10845).
