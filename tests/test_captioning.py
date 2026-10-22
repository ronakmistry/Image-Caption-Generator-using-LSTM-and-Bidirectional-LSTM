import numpy as np
import pytest
from captioning.data import vocabulary, encode_caption, validate_splits, batches, load_flickr
from captioning.evaluation import generate_caption, evaluate
from captioning.pipeline import run


def test_padding_unknown_and_remainder():
    vocab = vocabulary(['a square'])
    assert vocab['<pad>'] == 0
    assert encode_caption('unseen', vocab)[1] == vocab['<unk>']
    batch_list = list(batches(['x'], {'x': ['a square']}, {'x': np.zeros(4)}, vocab, 4, 2))
    assert [len(y) for _, y in batch_list] == [2, 1]
    assert batch_list[0][0][1][0].tolist() == [vocab['<start>'], 0, 0, 0]


def test_split_leakage_and_missing_resources(tmp_path):
    with pytest.raises(ValueError, match='overlapping'):
        validate_splits({'train': ['same'], 'test': ['same']})
    with pytest.raises(FileNotFoundError, match='Supply Flickr8k'):
        load_flickr(tmp_path)


class FixedModel:
    def __init__(self, index):
        self.index, self.calls = index, 0
    def __call__(self, inputs, training=False):
        self.calls += 1
        p = np.zeros((1, 6)); p[0, self.index] = 1
        return p


def test_decoding_and_explicit_evaluation_model():
    vocab = vocabulary(['a square'])
    end = FixedModel(vocab['<end>'])
    word = FixedModel(vocab['a'])
    assert generate_caption(end, np.zeros(4), vocab, 4) == ''
    assert generate_caption(word, np.zeros(4), vocab, 4) == 'a a a'
    assert word.calls == 3
    evaluate(end, ['x'], {'x': ['a square']}, {'x': np.zeros(4)}, vocab, 4)
    assert end.calls == 2 and word.calls == 3
    perfect = FixedModel(vocab['a'])
    report = evaluate(perfect, ['x'], {'x': ['a a a']}, {'x': np.zeros(4)}, vocab, 4)
    assert report['metrics']['mean_sentence_bleu_1'] == 1
    with pytest.raises(ValueError, match='empty'):
        evaluate(end, [], {}, {}, vocab, 4)


@pytest.mark.parametrize('encoder', ['mobilenet', 'inception'])
@pytest.mark.parametrize('variant', ['lstm', 'blstm'])
def test_full_fixture_round_trip(tmp_path, encoder, variant):
    report = run(tmp_path, encoder, variant)
    assert report['configuration']['feature_dim'] == (1024 if encoder == 'mobilenet' else 2048)
    assert report['image_count'] == 1
    assert np.isfinite(report['history'][0]['train']['loss'])
    assert 'fixture-only' in report['status']


def test_epoch_metrics_weight_partial_batches():
    from captioning.pipeline import train_epoch
    class BatchModel:
        def reset_metrics(self):
            pass
        def train_on_batch(self, inputs, labels, return_dict):
            return {'loss': float(np.mean(labels))}
        test_on_batch = train_on_batch
    batches = [(None, np.array([1, 3])), (None, np.array([8]))]
    assert train_epoch(BatchModel(), iter(batches), True)['loss'] == 4
    assert train_epoch(BatchModel(), iter(batches), False)['loss'] == 4
