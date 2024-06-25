import itertools

import Levenshtein
import numpy as np


def word_accuracy(gold_tokens, pred_tokens, train_vocab_tokens=None, orig_tokens=None):
    total_word_acc = np.mean([gold == pred for gold, pred in zip(gold_tokens, pred_tokens)])
    if train_vocab_tokens is None:
        return {'overall': total_word_acc}
    else:
        invocab_word_acc = np.mean([gold == pred for gold, pred, orig in zip(gold_tokens, pred_tokens, orig_tokens) if orig in train_vocab_tokens])
        oov_word_acc = np.mean([gold == pred for gold, pred, orig in zip(gold_tokens, pred_tokens, orig_tokens) if orig not in train_vocab_tokens])
        return {'overall': total_word_acc, 'invocab': invocab_word_acc, 'oov': oov_word_acc}

def cerI(gold_tokens, pred_tokens, train_vocab_tokens=None, orig_tokens=None):
    if orig_tokens is None:
        orig_tokens = []

    cerI_values = []
    cerI_invocab = []
    cerI_oov = []
    for gold, pred, orig in itertools.zip_longest(gold_tokens, pred_tokens, orig_tokens):
        if gold == pred:
            continue

        dist = Levenshtein.distance(gold, pred)
        cer = dist / max(1, len(gold))
        cerI_values.append(cer)

        if orig is not None:
            if orig in train_vocab_tokens:
                cerI_invocab.append(cer)
            else:
                cerI_oov.append(cer)


    if train_vocab_tokens is None:
        return {'overall': np.mean(cerI_values)}
    else:
        return {'overall': np.mean(cerI_values), 'invocab': np.mean(cerI_invocab), 'oov': np.mean(cerI_oov)}