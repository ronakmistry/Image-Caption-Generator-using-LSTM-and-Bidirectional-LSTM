"""Training and artifact round trips for fixture and real-data workflows."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
from PIL import Image
from .data import (load_flickr, validate_splits, vocabulary, batches,
                   preprocess_image, save_metadata)
from .models import build_encoder, build_decoder, keras, tf
from .evaluation import evaluate


def fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    captions = {}
    for i in range(4):
        key = f'fixture_{i}.jpg'
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype='uint8')).save(root / key)
        captions[key] = ['a small square', 'a colorful square']
    splits = {'train': ['fixture_0.jpg', 'fixture_1.jpg'],
              'validation': ['fixture_2.jpg'], 'test': ['fixture_3.jpg']}
    return captions, splits, root


def extract_features(ids, folder, encoder_name, weights):
    encoder = build_encoder(encoder_name, weights)
    return {key: np.asarray(encoder(preprocess_image(folder / key, encoder_name), training=False))[0]
            for key in ids}


def train_epoch(model, batch_iterator, training):
    model.reset_metrics()
    totals, count = {}, 0
    for inputs, labels in batch_iterator:
        call = model.train_on_batch if training else model.test_on_batch
        metrics = call(inputs, labels, return_dict=True)
        # Keras 3 resets its metric trackers inside each *_on_batch call.
        size = len(labels)
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * size
        count += size
    if not count:
        raise ValueError('No caption-prefix examples available')
    return {name: total / count for name, total in totals.items()}


def run(output='artifacts/smoke', encoder='mobilenet', variant='lstm', smoke=True,
        data_root=None, encoder_weights=None, epochs=15, batch_size=512, seed=42):
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    keras.utils.set_random_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if smoke:
        captions, splits, folder = fixture(output / 'fixture')
        epochs, batch_size = 1, 8
    else:
        if data_root is None:
            raise ValueError('Supply --data-root with Flickr8k files; see README')
        captions, splits, folder = load_flickr(data_root)
        if not encoder_weights or not Path(encoder_weights).is_file():
            raise FileNotFoundError('Supply --encoder-weights: local ImageNet no-top weights matching the encoder. No weights are downloaded.')
    validate_splits(splits)
    vocab = vocabulary([c for key in splits['train'] for c in captions[key]])
    max_len = max(len(c.split()) + 2 for key in splits['train'] for c in captions[key])
    ids = sorted(key for group in splits.values() for key in group)
    features = extract_features(ids, folder, encoder, None if smoke else encoder_weights)
    np.savez_compressed(output / 'features.npz', **features)
    feature_dim = len(features[ids[0]])
    config = {'encoder': encoder, 'variant': variant, 'feature_dim': feature_dim,
              'max_len': max_len, 'vocab_size': len(vocab), 'units': 8 if smoke else 256,
              'embedding_dim': 12 if smoke else 300, 'epochs': epochs,
              'batch_size': batch_size, 'seed': seed, 'preprocessing': 'RGB, resize, pixels/127.5-1',
              'encoder_weights': 'random fixture weights' if smoke else str(encoder_weights),
              'encoder_weights_sha256': None if smoke else hashlib.sha256(Path(encoder_weights).read_bytes()).hexdigest()}
    model = build_decoder(feature_dim, len(vocab), max_len, variant == 'blstm',
                          config['units'], config['embedding_dim'])
    history = []
    for epoch in range(epochs):
        # Deterministically shuffle image order each epoch; keep all captions per image together.
        train_ids = np.random.default_rng(seed + epoch).permutation(splits['train'])
        train = train_epoch(model, batches(train_ids, captions, features, vocab, max_len, batch_size), True)
        validation = train_epoch(model, batches(splits['validation'], captions, features, vocab, max_len, batch_size), False)
        history.append({'epoch': epoch + 1, 'train': train, 'validation': validation})
    metadata = {'configuration': config, 'vocabulary': vocab, 'splits': splits,
                'caption_sha256': hashlib.sha256(json.dumps(captions, sort_keys=True).encode()).hexdigest()}
    save_metadata(output / 'metadata.json', metadata)
    model.save(output / 'model.keras')
    restored = keras.models.load_model(output / 'model.keras')
    inputs, _ = next(batches(splits['test'], captions, features, vocab, max_len, batch_size))
    np.testing.assert_allclose(model(inputs, training=False), restored(inputs, training=False), atol=1e-6)
    report = evaluate(restored, splits['test'], captions, features, vocab, max_len)
    report.update({'status': 'fixture-only; not research evidence' if smoke else 'real-data rerun',
                   'configuration': config, 'history': history,
                   'environment': {'python': platform.python_version(), 'tensorflow': tf.__version__,
                                   'keras': keras.__version__, 'numpy': np.__version__}})
    save_metadata(output / 'results.json', report)
    return report


def predict(artifact_dir, image_path, encoder_weights):
    artifact_dir = Path(artifact_dir)
    metadata = json.loads((artifact_dir / 'metadata.json').read_text())
    config = metadata['configuration']
    if config['encoder_weights_sha256'] is None:
        raise ValueError('Fixture artifacts have random encoder weights; use a real-data trained artifact for external images')
    if hashlib.sha256(Path(encoder_weights).read_bytes()).hexdigest() != config['encoder_weights_sha256']:
        raise ValueError('Encoder weights do not match training metadata')
    encoder = build_encoder(config['encoder'], encoder_weights)
    feature = np.asarray(encoder(preprocess_image(image_path, config['encoder']), training=False))[0]
    decoding = metadata.get('decoding', {})
    if decoding.get('beam_width', 1) > 1:
        from .decoding import decode
        model = keras.models.load_model(artifact_dir / 'model.keras')

        @tf.function
        def infer(features, tokens):
            return model([features, tokens], training=False)

        return decode(infer, feature[None], metadata['vocabulary'], config['max_len'],
                      decoding['beam_width'], decoding.get('length_penalty', .6))[0]
    from .evaluation import generate_caption
    return generate_caption(keras.models.load_model(artifact_dir / 'model.keras'), feature,
                            metadata['vocabulary'], config['max_len'])
