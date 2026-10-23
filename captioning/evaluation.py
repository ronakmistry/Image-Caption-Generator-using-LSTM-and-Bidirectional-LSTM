"""Explicit model selection and documented BLEU variants."""
import warnings
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


def generate_caption(model, feature, vocab, max_len):
    inverse = {i: word for word, i in vocab.items()}
    prefix, result = [vocab['<start>']], []
    for _ in range(max_len - 1):
        tokens = np.zeros((1, max_len), dtype='int32')
        tokens[0, :len(prefix)] = prefix
        probabilities = np.asarray(model([np.asarray(feature)[None], tokens], training=False))[0].copy()
        probabilities[[vocab['<pad>'], vocab['<start>']]] = -np.inf
        word = int(np.argmax(probabilities))
        if word == vocab['<end>']:
            break
        result.append(inverse[word])
        prefix.append(word)
    return ' '.join(result)


def evaluate(model, ids, captions, features, vocab, max_len):
    if not ids:
        raise ValueError('Cannot evaluate an empty image set')
    predictions = {key: generate_caption(model, features[key], vocab, max_len) for key in ids}
    references = [[caption.split() for caption in captions[key]] for key in ids]
    hypotheses = [predictions[key].split() for key in ids]
    # No downloaded tokenizer resources and no smoothing. Scores remain on [0, 1].
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        mean_sentence = float(np.mean([sentence_bleu(r, h, weights=(1, 0, 0, 0))
                                       for r, h in zip(references, hypotheses)]))
        metrics = {'mean_sentence_bleu_1': mean_sentence}
        for n in range(1, 5):
            metrics[f'corpus_bleu_{n}'] = float(corpus_bleu(references, hypotheses, weights=(1/n,) * n))
    return {'metrics': metrics, 'predictions': predictions, 'image_count': len(ids)}
