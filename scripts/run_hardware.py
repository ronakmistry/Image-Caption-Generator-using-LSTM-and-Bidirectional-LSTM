"""Run the original MobileNet/LSTM on local Flickr8k with hardware timings.

Data and weight acquisition is a separate explicit step. No implicit downloads.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', str(__import__('pathlib').Path('.cache/matplotlib').resolve()))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import platform
import time
import numpy as np
from captioning.data import load_flickr, vocabulary, encode_caption, preprocess_image
from captioning.models import build_encoder, build_decoder, keras, tf
from captioning.evaluation import generate_caption

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--benchmark-only', action='store_true')
parser.add_argument('--output', default='artifacts/hardware-run')
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()
if args.device == 'cpu':
    tf.config.set_visible_devices([], 'GPU')
elif not tf.config.list_physical_devices('GPU'):
    raise SystemExit('No GPU detected. Install the separate Metal runtime or select --device cpu.')
keras.utils.set_random_seed(42)
output = Path(args.output); output.mkdir(parents=True, exist_ok=True)

def write(name, value):
    temporary = output / (name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(output / name)

def log(message):
    print(message, flush=True)

data_root = Path('data/flickr8k')
weights = data_root / 'downloads/mobilenet_1_0_224_tf_no_top.h5'
captions, splits, folder = load_flickr(data_root)
ids = sorted(key for group in splits.values() for key in group)
vocab = vocabulary([c for key in splits['train'] for c in captions[key]])
max_len = max(len(c.split()) + 2 for key in splits['train'] for c in captions[key])
metadata = {'configuration': {'encoder': 'mobilenet', 'variant': 'lstm', 'feature_dim': 1024,
    'vocab_size': len(vocab), 'max_len': max_len, 'units': 256, 'embedding_dim': 300,
    'seed': 42, 'batch_size': args.batch_size, 'preprocessing': 'RGB, resize, pixels/127.5-1',
    'encoder_weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
    'encoder_weights': str(weights)}, 'vocabulary': vocab, 'splits': splits,
    'caption_sha256': hashlib.sha256(json.dumps(captions, sort_keys=True).encode()).hexdigest()}
if not args.benchmark_only:
    write('metadata.json', metadata)
log(f'Device: {args.device}; train/validation/test images: {[len(s) for s in splits.values()]}; vocab={len(vocab)}; max_len={max_len}')
cache = data_root / 'mobilenet-features.npz'
cache_meta = cache.with_suffix('.json')
cache_identity = {'encoder_weights_sha256': metadata['configuration']['encoder_weights_sha256'],
                  'ids': ids, 'preprocessing': metadata['configuration']['preprocessing']}
if cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text())['identity'] == cache_identity:
    with np.load(cache, allow_pickle=False) as saved:
        features = saved['features']
    feature_timing = json.loads(cache_meta.read_text())['timing']
    log('Loaded cached pretrained MobileNet features')
else:
    encoder = build_encoder('mobilenet', str(weights))
    features = np.empty((len(ids), 1024), dtype='float32')
    start = time.perf_counter()
    for first in range(0, len(ids), 64):
        group = ids[first:first+64]
        pixels = np.concatenate([preprocess_image(folder / key, 'mobilenet') for key in group])
        features[first:first+len(group)] = np.asarray(encoder(pixels, training=False))
        if first % 512 == 0:
            log(f'Feature extraction: {first+len(group)}/{len(ids)} images; {time.perf_counter()-start:.1f}s')
    elapsed = time.perf_counter() - start
    feature_timing = {'seconds': elapsed, 'images_per_second': len(ids)/elapsed, 'device': args.device}
    np.savez_compressed(cache, features=features)
    cache_meta.write_text(json.dumps({'identity': cache_identity, 'timing': feature_timing}, indent=2))
    log(f'Feature extraction complete: {feature_timing}')
    del encoder
lookup = {key: i for i, key in enumerate(ids)}

def prepare(group):
    image_indices, sequences, labels = [], [], []
    for key in group:
        for caption in captions[key]:
            encoded = encode_caption(caption, vocab)
            for end in range(1, len(encoded)):
                prefix = np.zeros(max_len, dtype='int32')
                tokens = encoded[:end][-max_len:]
                prefix[:len(tokens)] = tokens
                sequences.append(prefix); labels.append(encoded[end]); image_indices.append(lookup[key])
    return np.asarray(image_indices), np.asarray(sequences), np.asarray(labels, dtype='int32')

train = prepare(splits['train'])
validation = prepare(splits['validation'])
log(f'Training caption-prefix examples: {len(train[2])}; validation examples: {len(validation[2])}')
model = build_decoder(1024, len(vocab), max_len)
if args.resume:
    model = keras.models.load_model(output / 'model.keras')
    history = json.loads((output / 'history.json').read_text())
    for record in history:
        record.setdefault('metric_scope', 'last batch only; initial runtime before summary correction')
    write('history.json', history)
else:
    history = []

def batch(data, indices):
    image_indices, sequences, labels = data
    return [features[image_indices[indices]], sequences[indices]], labels[indices]

inputs, targets = batch(train, np.arange(min(args.batch_size, len(train[2]))))
log('Compiling and warming up one full-width training batch ...')
model.train_on_batch(inputs, targets)
timings = []
for _ in range(3):
    start = time.perf_counter()
    metrics = model.train_on_batch(inputs, targets, return_dict=True)
    timings.append(time.perf_counter() - start)
bench = {'device': args.device, 'batch_size': args.batch_size,
         'median_batch_seconds': float(np.median(timings)), 'batch_seconds': timings,
         'examples_per_second': len(targets) / float(np.median(timings)),
         'estimated_epoch_seconds': int(np.ceil(len(train[2]) / args.batch_size)) * float(np.median(timings)),
         'feature_extraction': feature_timing,
         'environment': {'python': platform.python_version(), 'tensorflow': tf.__version__, 'keras': keras.__version__,
                         'physical_devices': [str(d) for d in tf.config.list_physical_devices()]}}
write(f'benchmark-{args.device}.json', bench)
log('Benchmark: ' + json.dumps(bench, indent=2))
if args.benchmark_only:
    raise SystemExit(0)
# Benchmark updates were only warmup; start training from a fresh seeded model.
if args.resume:
    model = keras.models.load_model(output / 'model.keras')
else:
    keras.utils.set_random_seed(42)
    model = build_decoder(1024, len(vocab), max_len)

preview_ids = list(np.random.default_rng(42).choice(splits['validation'], size=6, replace=False))

def preview(epoch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import textwrap
    from PIL import Image
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    predictions = {}
    for ax, key in zip(axes.flat, preview_ids):
        predicted = generate_caption(model, features[lookup[key]], vocab, max_len)
        predictions[key] = {'generated': predicted, 'reference': captions[key][0]}
        ax.imshow(Image.open(folder / key)); ax.axis('off')
        ax.set_title(textwrap.fill(predicted or '(empty caption)', 44), fontsize=11)
    fig.suptitle(f'Original MobileNet/LSTM · Flickr8k validation images · epoch {epoch}\nTrained locally; early captions may be repetitive or wrong', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output / 'caption-preview.png', dpi=140)
    plt.close(fig)
    write('preview.json', {'epoch': epoch, 'predictions': predictions})
    log('Preview: ' + json.dumps(predictions))

for epoch in range(len(history), args.epochs):
    order = np.random.default_rng(42 + epoch).permutation(len(train[2]))
    model.reset_metrics()
    start, last = time.perf_counter(), time.perf_counter()
    totals, count = {}, 0
    for first in range(0, len(order), args.batch_size):
        inputs, targets = batch(train, order[first:first+args.batch_size])
        metrics = model.train_on_batch(inputs, targets, return_dict=True)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * len(targets)
        count += len(targets)
        if time.perf_counter() - last > 20:
            done = min(first+args.batch_size,len(order))
            progress = {'epoch': epoch+1, 'epochs_requested': args.epochs, 'examples_done': done,
                        'examples_total': len(order), 'elapsed_seconds': time.perf_counter()-start,
                        'train': {k:v/count for k,v in totals.items()}}
            write('progress.json', progress)
            log('Training: '+json.dumps(progress));last=time.perf_counter()
    train_metrics = {k: v/count for k,v in totals.items()}
    training_seconds = time.perf_counter()-start
    model.reset_metrics()
    totals, count = {}, 0
    for first in range(0, len(validation[2]), args.batch_size):
        inputs, targets = batch(validation, slice(first, first+args.batch_size))
        metrics = model.test_on_batch(inputs, targets, return_dict=True)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * len(targets)
        count += len(targets)
    record = {'epoch':epoch+1, 'training_seconds':training_seconds,
              'epoch_seconds':time.perf_counter()-start, 'train':train_metrics,
              'validation':{k:v/count for k,v in totals.items()}, 'metric_scope':'weighted epoch mean'}
    history.append(record); write('history.json',history)
    model.save(output / 'model.keras')
    log('Epoch complete: '+json.dumps(record))
    preview(epoch+1)
write('complete.json', {'epochs': len(history), 'hardware_benchmark': bench, 'history': history,
                        'status':'Real Flickr8k training; test set not used for model selection',
                        'dataset_provenance':json.loads((data_root/'provenance.json').read_text())})
log('Training complete. Model, metadata, timings and preview saved to '+str(output))
