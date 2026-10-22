"""Evaluate the final trained checkpoint on the untouched Flickr8k test split."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--validation-preview', action='store_true')
args=parser.parse_args()
import time
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from captioning.data import load_flickr

output = Path('artifacts/hardware-run')
metadata = json.loads((output/'metadata.json').read_text())
config, vocab = metadata['configuration'], metadata['vocabulary']
inverse = {v:k for k,v in vocab.items()}
captions, splits, folder = load_flickr('data/flickr8k')
ids = sorted(key for group in splits.values() for key in group)
lookup = {key:i for i,key in enumerate(ids)}
with np.load('data/flickr8k/mobilenet-features.npz', allow_pickle=False) as data:
    features = data['features']
model = keras.models.load_model(output/'model.keras')

@tf.function(input_signature=[tf.TensorSpec([None,1024],tf.float32),tf.TensorSpec([None,config['max_len']],tf.int32)])
def infer(features, tokens):
    return model([features,tokens],training=False)

start = time.perf_counter()
group_ids = list(json.loads((output/'preview.json').read_text())['predictions']) if args.validation_preview else splits['test']
predictions = {}
for first in range(0,len(group_ids),64):
    group = group_ids[first:first+64]
    image_features = features[[lookup[key] for key in group]]
    tokens = np.zeros((len(group),config['max_len']),dtype='int32')
    tokens[:,0] = vocab['<start>']
    words = [[] for _ in group]
    active = np.ones(len(group),dtype=bool)
    for index in range(1,config['max_len']):
        scores = np.asarray(infer(image_features,tokens)).copy()
        scores[:,[vocab['<pad>'],vocab['<start>']]] = -np.inf
        selected = scores.argmax(axis=1)
        for row,word in enumerate(selected):
            if active[row] and word != vocab['<end>']:
                words[row].append(inverse[int(word)])
        active &= selected != vocab['<end>']
        tokens[:,index] = np.where(active,selected,0)
        if not active.any():break
    predictions.update({key:' '.join(text) for key,text in zip(group,words)})
    print(f'Test captions: {min(first+64,len(group_ids))}/{len(group_ids)}',flush=True)
refs=[[c.split() for c in captions[key]] for key in group_ids]
hypotheses=[predictions[key].split() for key in group_ids]
with warnings.catch_warnings():
    warnings.simplefilter('ignore',UserWarning)
    metrics={'mean_sentence_bleu_1':float(np.mean([sentence_bleu(r,h,weights=(1,0,0,0)) for r,h in zip(refs,hypotheses)]))}
    for n in range(1,5):metrics[f'corpus_bleu_{n}']=float(corpus_bleu(refs,hypotheses,weights=(1/n,)*n))
device = 'metal' if tf.config.list_physical_devices('GPU') else 'cpu'
report={'device':device, 'metrics':metrics,'image_count':len(predictions),'evaluation_seconds':time.perf_counter()-start,
        'predictions':predictions,'tokenization':'whitespace, case preserved, no smoothing',
        'status':('validation preview; not a benchmark' if args.validation_preview else 'real Flickr8k test evaluation; one final checkpoint, no test-set tuning')}
(output/(f'preview-compiled-{device}.json' if args.validation_preview else 'test-results.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='predictions'},indent=2),flush=True)

if args.validation_preview:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import textwrap
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (key, caption) in zip(axes.flat, predictions.items()):
        ax.imshow(Image.open(folder / key))
        ax.axis('off')
        ax.set_title(textwrap.fill(caption or '(empty caption)', 48), fontsize=11)
    epoch = len(json.loads((output / 'history.json').read_text()))
    figure.suptitle(f'MobileNet + LSTM · {epoch} epochs on Apple M5\nFixed validation examples · compiled {device} inference', fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, .93))
    figure.savefig(output / f'caption-preview-{device}.png', dpi=140)
    plt.close(figure)
