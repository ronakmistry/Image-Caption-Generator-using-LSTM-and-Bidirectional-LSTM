"""Save reproducibility probes for the selected improved checkpoint."""
from pathlib import Path
import os,sys,argparse,json
os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,tensorflow as tf,keras
import captioning.portable
from captioning.decoding import decode
parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','metal'],required=True);args=parser.parse_args()
if args.device=='cpu':tf.config.set_visible_devices([], 'GPU')
p=Path('artifacts/improvement-run');meta=json.loads((p/'metadata.json').read_text());v=meta['vocabulary'];max_len=meta['configuration']['max_len']
model=keras.models.load_model(p/'best.keras')
ids=sorted(key for group in meta['splits'].values() for key in group);lookup={key:i for i,key in enumerate(ids)}
features=np.load('data/flickr8k/mobilenet-features.npz')['features']
preview_ids=list(json.loads((p/'preview.json').read_text()))
f=features[[lookup[key] for key in preview_ids]]
tokens=np.zeros((6,max_len),dtype='int32')
for i in range(6):tokens[i,:i+1]=[v['<start>']]+[v['dog']]*i
@tf.function(input_signature=[tf.TensorSpec([None,1024],tf.float32),tf.TensorSpec([None,max_len],tf.int32)])
def infer(f,t):return model([f,t],training=False)
eager=np.asarray(model([f,tokens],training=False));graph=np.asarray(infer(f,tokens))
np.testing.assert_allclose(eager,graph,rtol=2e-4,atol=2e-5)
np.savez(p/f'consistency-{args.device}.npz',eager=eager,graph=graph)
captions=decode(infer,f,v,max_len,3)
expected=json.loads((p/'preview.json').read_text())
assert captions==[expected[key]['improved'] for key in preview_ids]
(p/f'consistency-{args.device}.json').write_text(json.dumps({'captions':dict(zip(preview_ids,captions)),'eager_graph_max_abs_difference':float(np.max(np.abs(eager-graph)))},indent=2)+'\n')
print(args.device,'saved checkpoint reproduces selected captions; max eager/graph difference',np.max(np.abs(eager-graph)))
