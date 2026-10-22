import numpy as np
from captioning.decoding import decode,bleu_metrics


def test_greedy_and_beam_select_different_complete_sequences():
    vocab={'<pad>':0,'<unk>':1,'<start>':2,'<end>':3,'a':4,'b':5}
    def infer(features,tokens):
        result=np.zeros((len(tokens),6),dtype='float32')
        for index,row in enumerate(tokens):
            length=np.count_nonzero(row)
            if length==1:result[index,4:]=[.55,.45]
            elif row[1]==5:result[index,3]=.9;result[index,4]=.1
            elif length==2:result[index,3]=.4;result[index,5]=.6
            else:result[index,3]=1
        return result
    features=np.zeros((2,4),dtype='float32')
    assert decode(infer,features,vocab,5,1)==['a b','a b']
    assert decode(infer,features,vocab,5,3)==['b','b']


def test_repeat_rate_and_perfect_bleu():
    result=bleu_metrics([['a b c d'],['a a a a']],['a b c d','a a a a'])
    assert result['corpus_bleu_4']==1
    assert result['adjacent_repeat_rate']==.5
