"""Compare identical decoder inputs across execution modes without training."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse,json,hashlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
parser=argparse.ArgumentParser()
parser.add_argument('--device',choices=['cpu','metal'],required=True)
args=parser.parse_args()
if args.device=='cpu':tf.config.set_visible_devices([], 'GPU')
root=Path('artifacts/hardware-run');meta=json.loads((root/'metadata.json').read_text());v=meta['vocabulary']
m=keras.models.load_model(root/'model.keras',compile=False)
ids=sorted(k for group in meta['splits'].values() for k in group);key=list(json.loads((root/'preview.json').read_text())['predictions'])[0]
f=np.load('data/flickr8k/mobilenet-features.npz')['features'][ids.index(key)]
weights_hash=hashlib.sha256(b''.join(np.asarray(w).tobytes() for w in m.weights)).hexdigest()
outputs={};summary={'device':args.device,'weights_sha256':weights_hash,'layers':[(l.name,l.__class__.__name__) for l in m.layers]}
@tf.function(input_signature=[tf.TensorSpec([None,1024],tf.float32),tf.TensorSpec([None,40],tf.int32)])
def call(f,t):return m([f,t],training=False)
for size in [1,6,64]:
 features=np.repeat(f[None],size,axis=0)
 tokens=np.zeros((size,40),dtype='int32')
 tokens[:,:3]=[v['<start>'],v['A'],v['dog']]
 for mode in ['eager','graph']:
  pred=np.asarray(m([features,tokens],training=False) if mode=='eager' else call(features,tokens)).copy()
  outputs[f'{mode}_{size}']=pred[0]
  summary[f'{mode}_{size}']={'top_ids':pred[0].argsort()[-5:][::-1].tolist(), 'row_max_diff':float(np.max(np.abs(pred-pred[:1])))}
np.savez(root/f'inference-probe-{args.device}.npz',**outputs)
(root/f'inference-probe-{args.device}.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
