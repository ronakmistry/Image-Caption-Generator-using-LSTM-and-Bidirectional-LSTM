"""Train the historical BLSTM alternative and compare on validation only.

Commands: --probe cpu, --probe metal, --train, --verify cpu, --verify metal.
Run CPU probes first; Metal commands require the local Metal environment.
"""
from pathlib import Path
import os, sys, argparse, json, time, hashlib
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tensorflow as tf
import keras
from captioning.models import build_decoder
from captioning.portable import gathered_clone
from captioning.data import load_flickr, encode_caption
from captioning.decoding import decode, bleu_metrics

parser = argparse.ArgumentParser()
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument('--probe', choices=['cpu', 'metal'])
mode.add_argument('--train', action='store_true')
mode.add_argument('--verify', choices=['cpu', 'metal'])
mode.add_argument('--benchmark', choices=['cpu', 'metal'])
mode.add_argument('--evaluate-only', action='store_true', help='Finalize a saved best checkpoint after a time-budget stop')
parser.add_argument('--units', type=int, default=64)
parser.add_argument('--train-device', choices=['cpu', 'metal'], default='cpu')
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--time-budget-seconds', type=float, default=3000,
                    help='Stop after a completed epoch once this elapsed budget is reached; 0 disables')
parser.add_argument('--output', type=Path, default=Path('artifacts/blstm-comparison'))
args = parser.parse_args()
device = args.probe or args.verify or args.benchmark or args.train_device
if device == 'cpu':
    tf.config.set_visible_devices([], 'GPU')
elif not tf.config.list_physical_devices('GPU'):
    raise SystemExit('Metal GPU is required for this command.')
keras.utils.set_random_seed(42)
out = args.output
out.mkdir(parents=True, exist_ok=True)
source = Path('artifacts/improvement-run')
metadata = json.loads((source / 'metadata.json').read_text())
vocab = metadata['vocabulary']
max_len = metadata['configuration']['max_len']

def write(name, value):
    temporary = out / (name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(out / name)

if args.benchmark:
    model = gathered_clone(build_decoder(1024, len(vocab), max_len, bidirectional=True, units=args.units))
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    inputs = [np.random.default_rng(7).normal(size=(512, 1024)).astype('float32'), np.zeros((512, max_len), 'int32')]
    for row in range(512):
        inputs[1][row, :row % max_len + 1] = np.random.default_rng(row).integers(1, len(vocab), row % max_len + 1)
    times = []
    print('Full-size batch benchmark starting', flush=True)
    for step in range(3):
        tick = time.perf_counter()
        model.train_on_batch(inputs, np.full(512, 3, 'int32'))
        times.append(time.perf_counter() - tick)
        print('batch', step, 'seconds', times[-1], flush=True)
    write(f'benchmark-{device}-{args.units}.json', {'batch_size': 512, 'units_per_direction': args.units, 'seconds': times})
    raise SystemExit

if args.probe:
    if device == 'cpu':
        model = gathered_clone(build_decoder(1024, 37, max_len, bidirectional=True, units=8, embedding_dim=12))
        model.save(out / 'probe.keras')
    else:
        model = keras.models.load_model(out / 'probe.keras', compile=False)
    inputs = [np.random.default_rng(7).normal(size=(4, 1024)).astype('float32'), np.zeros((4, max_len), 'int32')]
    for row, length in enumerate([1, 2, 19, 40]):
        inputs[1][row, :length] = np.arange(length) % 35 + 1
    @tf.function
    def graph(features, tokens):
        with tf.GradientTape() as tape:
            result = model([features, tokens], training=False)
            loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(tf.constant([4, 5, 6, 7]), result))
        gradients = tape.gradient(loss, model.trainable_variables)
        return result, [tf.convert_to_tensor(g) for g in gradients]
    result, gradients = graph(*inputs)
    eager = np.asarray(model(inputs, training=False))
    np.testing.assert_allclose(result, eager, atol=2e-5, rtol=2e-4)
    arrays = dict(probabilities=np.asarray(result), **{f'g{i}': np.asarray(g) for i, g in enumerate(gradients)})
    np.savez(out / f'probe-{device}.npz', **arrays)
    if device == 'metal':
        cpu = np.load(out / 'probe-cpu.npz')
        for name, values in arrays.items():
            np.testing.assert_allclose(cpu[name], values, atol=2e-5, rtol=2e-4)
        write('probe-summary.json', {'passed': True, 'prefix_lengths': [1, 2, 19, 40],
              'max_absolute_differences': {name: float(np.max(np.abs(cpu[name] - values))) for name, values in arrays.items()}})
    print(device, 'BLSTM forward/gradient probe passed', flush=True)
    raise SystemExit

captions, splits, folder = load_flickr('data/flickr8k')
assert splits == metadata['splits']
ids = sorted(key for group in splits.values() for key in group)
lookup = {key: i for i, key in enumerate(ids)}
features = np.load('data/flickr8k/mobilenet-features.npz')['features']
original_preview = list(json.loads((source / 'preview.json').read_text()))
extra = np.random.default_rng(20261021).choice(sorted(set(splits['validation']) - set(original_preview)), 6, replace=False).tolist()
preview_keys = original_preview + extra

def inference(model):
    @tf.function(input_signature=[tf.TensorSpec([None, 1024], tf.float32), tf.TensorSpec([None, max_len], tf.int32)])
    def infer(f, t):
        return model([f, t], training=False)
    return infer

if args.verify:
    model = keras.models.load_model(out / 'model.keras', compile=False)
    infer = inference(model)
    f = features[[lookup[k] for k in preview_keys]]
    tokens = np.zeros((len(f), max_len), 'int32')
    for i in range(len(f)):
        length = [1, 2, 19, 40][i % 4]
        tokens[i, :length] = [vocab['<start>']] + [vocab['dog']] * (length - 1)
    eager, graph = np.asarray(model([f, tokens], training=False)), np.asarray(infer(f, tokens))
    np.testing.assert_allclose(eager, graph, atol=2e-5, rtol=2e-4)
    np.savez(out / f'consistency-{device}.npz', probabilities=graph)
    texts = decode(infer, f, vocab, max_len, 3)
    expected = json.loads((out / 'validation-beam3.json').read_text())['predictions']
    assert texts == [expected[k] for k in preview_keys]
    write(f'consistency-{device}.json', {'captions': dict(zip(preview_keys, texts)), 'passed': True})
    if (out / 'consistency-cpu.npz').exists() and (out / 'consistency-metal.npz').exists():
        cpu = np.load(out / 'consistency-cpu.npz')['probabilities']
        metal = np.load(out / 'consistency-metal.npz')['probabilities']
        np.testing.assert_allclose(cpu, metal, atol=2e-5, rtol=2e-4)
        write('consistency-summary.json', {'passed': True, 'images': len(f), 'max_absolute_difference': float(np.max(np.abs(cpu-metal)))})
    print(device, 'trained checkpoint probabilities and captions verified', flush=True)
    raise SystemExit

if (out / 'complete.json').exists():
    raise SystemExit('Completed comparison exists; refusing to overwrite it.')
if args.evaluate_only:
    history = json.loads((out / 'history.json').read_text())
    best_epoch = history[-1]['best_epoch']
    source_hash = json.loads((out / 'protocol.json').read_text())['baseline_sha256']
    started = time.perf_counter() - sum(row['seconds'] for row in history)
    protocol = json.loads((out / 'protocol.json').read_text())
    protocol['stop_reason'] = f'runtime budget; interrupted after {len(history)} completed epochs, before maximum fifteen'
    write('protocol.json', protocol)
else:
    history = None
if device == 'metal':
    assert json.loads((out / 'probe-summary.json').read_text())['passed']
else:
    assert (out / 'probe-cpu.npz').exists(), 'Run --probe cpu first'
if not args.evaluate_only:
    started = time.perf_counter()
    source_hash = hashlib.sha256((source / 'model.keras').read_bytes()).hexdigest()
    native = build_decoder(1024, len(vocab), max_len, bidirectional=True, units=args.units)
    model = gathered_clone(native)
    del native
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    write('protocol.json', {'seed': 42, 'architecture': f'original three-stage BLSTM, {args.units} units per direction',
      'initialization': 'random; no caption decoder weights transferred', 'maximum_epochs': args.epochs,
      'time_budget_seconds': args.time_budget_seconds,
      'batch_size': 512, 'checkpoint_selection': 'minimum validation cross entropy',
      'lr_factor': .5, 'lr_patience': 2, 'early_stop_patience': 4, 'min_delta': .005,
      'decoding': 'greedy and beam width 3, length penalty 0.6',
      'comparison': 'existing 23-epoch LSTM; unequal training budgets, exploratory comparison',
      'training_device': device, 'parameter_count': model.count_params(),
      'test_split': 'not evaluated', 'baseline_sha256': source_hash,
      'preview': {'existing_images': original_preview, 'extra_seed': 20261021, 'extra_images': extra}})

def prepare(group):
    image_indices, sequences, targets = [], [], []
    for key in group:
        for caption in captions[key]:
            encoded = encode_caption(caption, vocab)
            for end in range(1, len(encoded)):
                row = np.zeros(max_len, 'int32')
                prefix = encoded[:end][-max_len:]
                row[:len(prefix)] = prefix
                image_indices.append(lookup[key]); sequences.append(row); targets.append(encoded[end])
    return np.asarray(image_indices), np.asarray(sequences), np.asarray(targets, 'int32')

if not args.evaluate_only:
    train, validation = prepare(splits['train']), prepare(splits['validation'])

def pass_data(data, training, order=None):
    indices, sequences, targets = data
    order = np.arange(len(targets)) if order is None else order
    totals, count = {}, 0
    for first in range(0, len(order), 512):
        index = order[first:first+512]
        model.reset_metrics()  # Batch metrics must not be cumulative when weighting batches.
        call = model.train_on_batch if training else model.test_on_batch
        logs = call([features[indices[index]], sequences[index]], targets[index], return_dict=True)
        for key, value in logs.items():
            totals[key] = totals.get(key, 0) + float(value) * len(index)
        count += len(index)
    return {key: value / count for key, value in totals.items()}

if not args.evaluate_only:
    best, significant, wait, lr_wait, history, best_epoch = float('inf'), float('inf'), 0, 0, [], 0
    print('BLSTM training started; progress is saved locally without per-epoch notifications.', flush=True)
for epoch in ([] if args.evaluate_only else range(1, args.epochs + 1)):
    tick = time.perf_counter()
    lr = float(model.optimizer.learning_rate.numpy())
    trained = pass_data(train, True, np.random.default_rng(42 + epoch).permutation(len(train[2])))
    validated = pass_data(validation, False)
    if validated['loss'] < best:
        best, best_epoch = validated['loss'], epoch
        model.save(out / 'model.keras')
    if validated['loss'] < significant - .005:
        significant, wait, lr_wait = validated['loss'], 0, 0
    else:
        wait += 1; lr_wait += 1
        if lr_wait >= 2:
            model.optimizer.learning_rate.assign(max(lr * .5, 1e-5)); lr_wait = 0
    history.append({'epoch': epoch, 'train': trained, 'validation': validated,
                    'learning_rate': lr, 'seconds': time.perf_counter() - tick, 'best_epoch': best_epoch})
    write('history.json', history)
    if wait >= 4:
        break
    if args.time_budget_seconds and time.perf_counter() - started >= args.time_budget_seconds:
        protocol = json.loads((out / 'protocol.json').read_text())
        protocol['stop_reason'] = 'elapsed time budget reached after completed epoch'
        write('protocol.json', protocol)
        break
model = keras.models.load_model(out / 'model.keras', compile=False)
infer = inference(model)
results = {}
for name, width in [('greedy', 1), ('beam3', 3)]:
    predictions = {}
    for first in range(0, len(splits['validation']), 32):
        keys = splits['validation'][first:first+32]
        texts = decode(infer, features[[lookup[k] for k in keys]], vocab, max_len, width)
        predictions.update(zip(keys, texts))
    metrics = bleu_metrics([captions[k] for k in splits['validation']], [predictions[k] for k in splits['validation']])
    results[name] = metrics
    write(f'validation-{name}.json', {'metrics': metrics, 'predictions': predictions})
method = max(results, key=lambda key: results[key]['corpus_bleu_4'])
baseline = json.loads((source / 'validation-beam3.json').read_text())
candidate = json.loads((out / f'validation-{method}.json').read_text())
preview = {key: {'lstm': baseline['predictions'][key], 'blstm': candidate['predictions'][key],
                 'references': captions[key]} for key in preview_keys}
write('preview.json', preview)
metadata['configuration'].update(variant='blstm', epochs=best_epoch, units=args.units)
metadata['decoding'] = {'beam_width': 3 if method == 'beam3' else 1, 'length_penalty': .6}
metadata['experiment'] = json.loads((out / 'protocol.json').read_text())
write('metadata.json', metadata)
write('complete.json', {'best_epoch': best_epoch, 'epochs_run': len(history), 'decoding': method,
      'validation': results, 'baseline_validation': baseline['metrics'],
      'source_unchanged': source_hash == hashlib.sha256((source / 'model.keras').read_bytes()).hexdigest(),
      'elapsed_seconds': time.perf_counter()-started, 'test_evaluated': False})
print(json.dumps(json.loads((out / 'complete.json').read_text()), indent=2), flush=True)
