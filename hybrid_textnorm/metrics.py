import numpy as np


def word_accuracy(gold_tokens, pred_tokens, train_vocab_tokens=None, orig_tokens=None, equality=None):
    if equality is None:
        equality = lambda x,y: x==y
    total_word_acc = np.mean([equality(gold,pred) for gold, pred in zip(gold_tokens, pred_tokens)])
    if train_vocab_tokens is None:
        return total_word_acc
    else:
        invocab_word_acc = np.mean([equality(gold,pred) for gold, pred, orig in zip(gold_tokens, pred_tokens, orig_tokens) if orig in train_vocab_tokens])
        oov_word_acc = np.mean([equality(gold,pred) for gold, pred, orig in zip(gold_tokens, pred_tokens, orig_tokens) if orig not in train_vocab_tokens])
        return total_word_acc, invocab_word_acc, oov_word_acc

def metrics_summary(gold_tokens, pred_tokens, train_vocab_tokens=None, orig_tokens=None):
    trans = str.maketrans("", "", '░▁')

    output = {}
    for mapper_type in ['full', 'space_insensitive', 'case_insensitive', 'space_case_insensitive']:
        out = {}
        mapper = None
        if mapper_type == 'space_insensitive':
            mapper = lambda x: x.translate(trans)
        if mapper_type == 'space_insensitive':
            mapper = lambda x: x.lower()
        if mapper_type == 'space_case_insensitive':
            mapper = lambda x: x.lower().translate(trans)

        acc = word_accuracy(gold_tokens, pred_tokens, train_vocab_tokens, orig_tokens, mapper)
        if type(acc) == tuple:
            total_word_acc, invocab_word_acc, oov_word_acc = acc
            out['word_acc'] = total_word_acc
            out['word_acc_invocab'] = invocab_word_acc
            out['word_acc_oov'] = oov_word_acc
        else:
            out['word_acc'] = acc
        output[mapper_type] = out

    return output

