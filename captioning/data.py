"""Deterministic, image-level data handling; no implicit downloads."""
from pathlib import Path
import json
import numpy as np
from PIL import Image

SPECIAL = ['<pad>', '<unk>', '<start>', '<end>']


def load_flickr(root):
    root = Path(root)
    text = root / 'Flickr8k_text'
    required = [text / 'Flickr8k.token.txt'] + [
        text / f'Flickr_8k.{split}Images.txt' for split in ('train', 'dev', 'test')]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('Supply Flickr8k captions and official splits; missing: ' + ', '.join(missing))
    captions = {}
    for line in required[0].read_text().splitlines():
        key, caption = line.split('\t', 1)
        captions.setdefault(key.rsplit('#', 1)[0], []).append(caption.strip())
    splits = {}
    for name, path in zip(('train', 'validation', 'test'), required[1:]):
        ids = path.read_text().splitlines()
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f'{name} split must be nonempty and have unique image IDs')
        splits[name] = sorted(ids)
    validate_splits(splits)
    folder = root / 'Flicker8k_Dataset'
    for ids in splits.values():
        for key in ids:
            if not (folder / key).is_file() or not captions.get(key):
                raise FileNotFoundError(f'Missing image or caption: {key}; supply {folder} and captions')
    return captions, splits, folder


def validate_splits(splits):
    seen = set()
    for name, ids in splits.items():
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError(f'Empty, duplicated or overlapping image IDs in {name}')
        seen.update(ids)


def vocabulary(captions):
    words = sorted({w for caption in captions for w in caption.split()} - set(SPECIAL))
    return {word: i for i, word in enumerate(SPECIAL + words)}


def encode_caption(caption, vocab):
    return [vocab['<start>']] + [vocab.get(w, vocab['<unk>']) for w in caption.split()] + [vocab['<end>']]


def prefix_examples(ids, captions, features, vocab, max_len):
    for key in ids:
        for caption in captions[key]:
            encoded = encode_caption(caption, vocab)
            for end in range(1, len(encoded)):
                tokens = np.zeros(max_len, dtype='int32')
                prefix = encoded[:end][-max_len:]
                tokens[:len(prefix)] = prefix
                yield features[key], tokens, encoded[end]


def batches(ids, captions, features, vocab, max_len, batch_size):
    batch = []
    for item in prefix_examples(ids, captions, features, vocab, max_len):
        batch.append(item)
        if len(batch) == batch_size:
            yield pack(batch)
            batch = []
    if batch:
        yield pack(batch)


def pack(batch):
    features, tokens, targets = zip(*batch)
    return [np.asarray(features, dtype='float32'), np.asarray(tokens)], np.asarray(targets, dtype='int32')


def preprocess_image(path, encoder):
    size = 299 if encoder == 'inception' else 224
    with Image.open(path) as img:
        pixels = np.asarray(img.convert('RGB').resize((size, size)), dtype='float32')
    return pixels[None] / 127.5 - 1.0


def save_metadata(path, metadata):
    Path(path).write_text(json.dumps(metadata, indent=2) + '\n')
