"""Batched greedy/beam decoding with explicit length normalization."""
import warnings
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu


def decode(infer, features, vocab, max_len, beam_width=1, length_penalty=0.6):
    features = np.asarray(features, dtype='float32')
    size, width, vocabulary_size = len(features), beam_width, len(vocab)
    inverse = {index:word for word,index in vocab.items()}
    tokens = np.zeros((size,width,max_len), dtype='int32')
    tokens[:,:,0] = vocab['<start>']
    scores = np.full((size,width), -np.inf, dtype='float64');scores[:,0] = 0
    lengths = np.zeros((size,width),dtype='int32')
    finished = np.zeros((size,width),dtype=bool)
    repeated_features = np.repeat(features,width,axis=0)
    for position in range(1,max_len):
        probabilities = np.asarray(infer(repeated_features,tokens.reshape(-1,max_len))).reshape(size,width,vocabulary_size)
        logp = np.log(np.maximum(probabilities,1e-30)).astype('float64')
        logp[:,:,[vocab['<pad>'],vocab['<start>']]] = -np.inf
        logp[finished] = -np.inf
        ended_rows, ended_beams = np.where(finished)
        logp[ended_rows,ended_beams,vocab['<end>']] = 0
        candidate_scores = scores[:,:,None] + logp
        candidate_lengths = lengths + (~finished)
        normalization = ((5 + candidate_lengths) / 6.0) ** length_penalty
        ranking = candidate_scores / normalization[:,:,None]
        flat = ranking.reshape(size,-1)
        selected = np.argpartition(flat,-width,axis=1)[:,-width:]
        chosen_scores = np.take_along_axis(flat,selected,axis=1)
        order = np.argsort(chosen_scores,axis=1)[:,::-1]
        selected = np.take_along_axis(selected,order,axis=1)
        previous, words = selected // vocabulary_size, selected % vocabulary_size
        rows = np.arange(size)[:,None]
        previous_finished = finished[rows,previous]
        tokens = tokens[rows,previous].copy()
        tokens[:,:,position] = np.where(previous_finished,0,words)
        scores = candidate_scores.reshape(size,-1)[rows,selected]
        lengths = candidate_lengths[rows,previous]
        finished = previous_finished | (words == vocab['<end>'])
        if finished.all():break
    best = np.argmax(scores / (((5 + lengths) / 6.0) ** length_penalty),axis=1)
    captions=[]
    for row, index in enumerate(best):
        words=[]
        for token in tokens[row,index,1:]:
            if token in (0,vocab['<end>']):break
            words.append(inverse[int(token)])
        captions.append(' '.join(words))
    return captions


def bleu_metrics(references, hypotheses):
    refs=[[caption.split() for caption in group] for group in references]
    hyps=[caption.split() for caption in hypotheses]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        result={'mean_sentence_bleu_1':float(np.mean([sentence_bleu(r,h,weights=(1,0,0,0)) for r,h in zip(refs,hyps)]))}
        for n in range(1,5):result[f'corpus_bleu_{n}']=float(corpus_bleu(refs,hyps,weights=(1/n,)*n))
    result['adjacent_repeat_rate']=float(np.mean([sum(a==b for a,b in zip(h,h[1:]))/max(1,len(h)-1) for h in hyps]))
    return result
