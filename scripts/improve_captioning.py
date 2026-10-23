"""Validation-selected continuation of the preserved 15-epoch checkpoint."""
from pathlib import Path
import os,sys
os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time,hashlib
import numpy as np,tensorflow as tf,keras
from captioning.portable import gathered_clone
from captioning.data import load_flickr,encode_caption
from captioning.decoding import decode,bleu_metrics

source=Path('artifacts/hardware-run');out=Path('artifacts/improvement-run')
out.mkdir(parents=True,exist_ok=True)
if (out/'complete.json').exists():raise SystemExit('Experiment already completed; preserve it before starting a new run.')
if not tf.config.list_physical_devices('GPU'):raise SystemExit('Run with the Metal environment outside the sandbox.')
keras.utils.set_random_seed(42)
start_run=time.perf_counter()
metadata=json.loads((source/'metadata.json').read_text());vocab=metadata['vocabulary'];max_len=metadata['configuration']['max_len']
captions,splits,folder=load_flickr('data/flickr8k')
ids=sorted(key for group in splits.values() for key in group);lookup={key:i for i,key in enumerate(ids)}
features=np.load('data/flickr8k/mobilenet-features.npz')['features']

def write(name,value):
    temp=out/(name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(out/name)

def prepare(group):
    image_indices,sequences,targets=[],[],[]
    for key in group:
        for caption in captions[key]:
            encoded=encode_caption(caption,vocab)
            for end in range(1,len(encoded)):
                row=np.zeros(max_len,dtype='int32');prefix=encoded[:end][-max_len:];row[:len(prefix)]=prefix
                image_indices.append(lookup[key]);sequences.append(row);targets.append(encoded[end])
    return np.asarray(image_indices),np.asarray(sequences),np.asarray(targets,dtype='int32')

train,validation=prepare(splits['train']),prepare(splits['validation'])
original=keras.models.load_model(source/'model.keras')
model=gathered_clone(original)
model.compile(optimizer=keras.optimizers.deserialize(keras.optimizers.serialize(original.optimizer)),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.optimizer.build(model.trainable_variables)
assert len(original.optimizer.variables)==len(model.optimizer.variables)
for old,new in zip(original.optimizer.variables,model.optimizer.variables):
    assert old.shape==new.shape
    new.assign(old)
source_hash=hashlib.sha256((source/'model.keras').read_bytes()).hexdigest()
write('metadata.json',dict(metadata,experiment={'source_sha256':source_hash,'additional_epochs_max':10,'batch_size':512,
    'initial_learning_rate':float(model.optimizer.learning_rate.numpy()),'lr_factor':.5,'lr_patience':2,'early_stop_patience':4,
    'min_delta':.005,'minimum_lr':1e-5,'optimizer_state':'preserved from source checkpoint','runtime':'GatherLSTM + AddV2 Dense on Metal'}))
model.save(out/'baseline.keras')

@tf.function(input_signature=[tf.TensorSpec([None,1024],tf.float32),tf.TensorSpec([None,max_len],tf.int32)])
def infer(image_features,tokens):return model([image_features,tokens],training=False)

def pass_data(data,training,order=None):
    indices,sequences,targets=data
    if order is None:order=np.arange(len(targets))
    totals={};count=0
    call=model.train_on_batch if training else model.test_on_batch
    for first in range(0,len(order),512):
        index=order[first:first+512]
        metrics=call([features[indices[index]],sequences[index]],targets[index],return_dict=True)
        for key,value in metrics.items():totals[key]=totals.get(key,0)+float(value)*len(index)
        count+=len(index)
    return {key:value/count for key,value in totals.items()}

def predict_group(group,beam_width=1):
    texts=[]
    for first in range(0,len(group),32):
        keys=group[first:first+32]
        texts.extend(decode(infer,features[[lookup[k] for k in keys]],vocab,max_len,beam_width))
    return dict(zip(group,texts))

baseline_loss=pass_data(validation,False)
baseline_predictions=predict_group(splits['validation'])
baseline_metrics=bleu_metrics([captions[k] for k in splits['validation']],list(baseline_predictions.values()))
write('baseline-validation.json',{'loss':baseline_loss,'metrics':baseline_metrics,'predictions':baseline_predictions})
model.save(out/'best.keras')
best_loss=baseline_loss['loss'];significant_best=best_loss;wait=0;lr_wait=0;best_epoch=15;history=[]
print('Baseline established. Starting up to ten additional epochs with corrected execution.',flush=True)
for epoch in range(16,26):
    start=time.perf_counter();lr=float(model.optimizer.learning_rate.numpy())
    order=np.random.default_rng(42+epoch-1).permutation(len(train[2]))
    train_metrics=pass_data(train,True,order)
    val_metrics=pass_data(validation,False)
    if val_metrics['loss']<best_loss:
        best_loss=val_metrics['loss'];best_epoch=epoch;model.save(out/'best.keras')
    if val_metrics['loss']<significant_best-.005:
        significant_best=val_metrics['loss'];wait=0;lr_wait=0
    else:
        wait+=1;lr_wait+=1
        if lr_wait>=2:
            model.optimizer.learning_rate.assign(max(lr*.5,1e-5));lr_wait=0
    record={'epoch':epoch,'train':train_metrics,'validation':val_metrics,'learning_rate':lr,'seconds':time.perf_counter()-start,
            'best_epoch':best_epoch,'no_improvement_epochs':wait}
    history.append(record);write('history.json',history);model.save(out/'last.keras')
    if wait>=4:break
# Keep the same model object so the compiled inference function remains valid.
selected=keras.models.load_model(out/'best.keras');model.set_weights(selected.get_weights())
results={}
for name,width in [('greedy',1),('beam3',3)]:
    predicted=predict_group(splits['validation'],width)
    metrics=bleu_metrics([captions[k] for k in splits['validation']],list(predicted.values()))
    results[name]={'metrics':metrics,'predictions':predicted}
    write(f'validation-{name}.json',results[name])
method=max(results,key=lambda name:results[name]['metrics']['corpus_bleu_4'])
write('selection.json',{'checkpoint_epoch':best_epoch,'checkpoint_selection':'minimum validation loss',
    'decoding':method,'beam_width':3 if method=='beam3' else 1,'length_penalty':.6,
    'decoding_selection':'maximum validation corpus BLEU-4; fixed two-method comparison'})
# Publish the selected checkpoint through the normal prediction API.
import shutil
shutil.copy2(out/'best.keras',out/'model.keras')
metadata['configuration']['epochs']=best_epoch
metadata['decoding']={'beam_width':3 if method=='beam3' else 1,'length_penalty':.6}
write('metadata.json',dict(metadata,experiment=json.loads((out/'metadata.json').read_text())['experiment']))
# Exactly one final test evaluation, after all selection decisions are saved.
test_predictions=predict_group(splits['test'],3 if method=='beam3' else 1)
test_metrics=bleu_metrics([captions[k] for k in splits['test']],list(test_predictions.values()))
write('test-results.json',{'metrics':test_metrics,'predictions':test_predictions,'image_count':1000,'selection':json.loads((out/'selection.json').read_text())})
preview_keys=list(json.loads((source/'preview.json').read_text())['predictions'])
preview={k:{'baseline':baseline_predictions[k],'improved':results[method]['predictions'][k],'reference':captions[k][0]} for k in preview_keys}
write('preview.json',preview)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import textwrap
fig,axes=plt.subplots(2,3,figsize=(15,9))
for ax,(key,row) in zip(axes.flat,preview.items()):
    ax.imshow(Image.open(folder/key));ax.axis('off');ax.set_title(textwrap.fill(row['improved'] or '(empty caption)',48),fontsize=11)
fig.suptitle(f'Corrected MobileNet/LSTM · checkpoint epoch {best_epoch} · {method}\nSame fixed validation images; model and decoding selected on validation only',fontsize=15)
fig.tight_layout(rect=(0,0,1,.93));fig.savefig(out/'caption-preview.png',dpi=140);plt.close(fig)
write('complete.json',{'source_sha256':source_hash,'source_unchanged':source_hash==hashlib.sha256((source/'model.keras').read_bytes()).hexdigest(),
    'additional_epochs':len(history),'best_epoch':best_epoch,'baseline_validation':baseline_metrics,
    'final_validation':{name:row['metrics'] for name,row in results.items()},'chosen_decoding':method,
    'test':test_metrics,'elapsed_seconds':time.perf_counter()-start_run,'history':history})
print(json.dumps(json.loads((out/'complete.json').read_text()),indent=2),flush=True)
