from pathlib import Path

import sys

import itertools

import Levenshtein
import pandas
from datasets import load_dataset, Dataset
from tqdm import tqdm
import regex as re

from hybrid_textnorm.align_levenshtein import align_token_sequences
from hybrid_textnorm.lexicon import Lexicon


def align_pred_sent_with_gold_sent(predicted_sent, gold_sent):
    original_spacing_tokens = []
    trans = str.maketrans("", "", '░▁')
    for gold_tok in gold_sent:
        original_spacing_tokens.append(gold_tok.translate(trans))

    alignment = align_token_sequences(original_spacing_tokens, predicted_sent)

    new_pred_sent = []
    for i in range(len(alignment)):
        if i < len(alignment) - 1 and re.match(r'\p{Punct}', alignment[i + 1][0]):
            new_pred_sent.append(alignment[i][1].replace('░', ''))
        else:
            new_pred_sent.append(alignment[i][1])

    return new_pred_sent


def read_file_sentences(filename, dataset_token_field=None, align='auto', gold_sentences=None):
    if filename.endswith('.jsonl'):
        ds = load_dataset('json', data_files=filename, split='train')
        for row in ds:
            yield row['tokens'][dataset_token_field]
    else:
        with open(filename) as f:
            lines = f.readlines()

        if all(' ' not in l for l in lines):
            # assume one token per line -> one big "sentence"
            yield [line.strip() for line in lines]
        else:
            # assume one (space-tokenized) sentence per line
            do_align = False
            if align == 'always':
                do_align = True
            elif align == 'auto':
                if 'transnormer' in filename or 'csmtiser_sent' in filename:
                    do_align = True

            if do_align:
                assert len(lines) == len(gold_sentences)
                ds = []
                for line, gold_sent in zip(lines, gold_sentences):
                    predicted_sent = line.strip().split(' ')
                    ds.append({'predicted_sent': predicted_sent, 'gold_sent': gold_sent})

                ds = Dataset.from_list(ds)
                ds = ds.map(lambda x: {'aligned': align_pred_sent_with_gold_sent(x['predicted_sent'], x['gold_sent'])},
                            num_proc=4)

                for row in ds:
                    aligned = row['aligned']
                    gold_sent = row['gold_sent']
                    assert len(aligned) == len(gold_sent)
                    yield aligned
            else:
                for line in lines:
                    yield line.strip().split(' ')

def besttype_prediction(orig_tokens, gold_tokens):
    lexicon = Lexicon.sequences_to_lexicon([{'orig': orig_tokens, 'norm': gold_tokens}])

    output = []
    for orig_tok in orig_tokens:
        output.append(lexicon[orig_tok].most_common()[0][0])

    return output

def lexicon_prediction(orig_tokens, train_lexicon):
    output = []
    for orig_token in orig_tokens:
        if orig_token in train_lexicon.keys():
            subst, _ = train_lexicon[orig_token].most_common(1)[0]
            output.append(subst)
        else:
            output.append(orig_token)

    return output

def get_error_features(pred_tokens, orig_tokens, gold_tokens, vocab):
    error_features = pandas.DataFrame(index=range(len(gold_tokens)),
                                      columns=['oov', 'correct', 'spacing', 'casing', 'character'], data=False)
    for i, (pred, orig, gold) in enumerate(zip(tqdm(pred_tokens), orig_tokens, gold_tokens)):
        # skip punctuation
        if re.fullmatch(r'\p{Punct}+', gold):
            pred = gold

        error_features.loc[i, 'oov'] = orig not in vocab
        error_features.loc[i, 'correct'] = pred == gold

        # first fix spacing / characters
        e = Levenshtein.editops(pred.lower(), gold.lower())
        for op, spos, dpos in e:
            if op == 'insert' and gold[dpos] in {'░','▁'}:
                error_features.loc[i, 'spacing'] = True
            elif op == 'delete' and pred[spos] in {'░','▁'}:
                error_features.loc[i, 'spacing'] = True
            elif op == 'replace' and (pred[spos] in {'░', '▁'} or gold[dpos] in {'░', '▁'}):
                error_features.loc[i, 'character'] = True
                error_features.loc[i, 'spacing'] = True
            else:
                error_features.loc[i, 'character'] = True

        # then fix casing
        pred = Levenshtein.apply_edit(e, pred, gold)
        e = Levenshtein.editops(pred, gold)
        for (op, spos, dpos) in e:
            assert not (pred[spos] in {'░', '▁'} or gold[dpos] in {'░', '▁'})
            if op == 'replace' and pred[spos].lower() == gold[dpos].lower():
                error_features.loc[i, 'casing'] = True
            else:
                error_features.loc[i, 'character'] = True

    return error_features

def accumulate_errors(pred_tokens, orig_tokens, gold_tokens, vocab):
    error_features = get_error_features(pred_tokens, orig_tokens, gold_tokens, vocab)
    accumulated = error_features[~error_features.correct].groupby('oov')[['spacing', 'casing', 'character']].sum().unstack().reorder_levels([1,0]).sort_index()
    accumulated.index = accumulated.index.map(lambda x: ('oov' if x[0] else 'invocab', x[1]))

    accumulated.loc[('oov', 'total')] = len(error_features[(~error_features.correct)&(error_features.oov)])
    accumulated.loc[('invocab', 'total')] = len(error_features[(~error_features.correct)&(~error_features.oov)])
    return accumulated

def main():
    output_df = pandas.DataFrame(columns=pandas.MultiIndex.from_product([['invocab', 'oov'], ['total', 'spacing', 'casing', 'character']]))

    train_lexicon = Lexicon.from_dataset('json', data_files='dataset/processed/train.lexicon.jsonl', split='train')
    vocab = train_lexicon.keys()
    gold_sentences = list(read_file_sentences('dataset/processed/test.jsonl', dataset_token_field='norm'))
    orig_sentences = list(read_file_sentences('dataset/processed/test.jsonl', dataset_token_field='orig'))

    gold_tokens = list(itertools.chain.from_iterable(gold_sentences))
    orig_tokens = list(itertools.chain.from_iterable(orig_sentences))

    # identity
    pred_tokens = orig_tokens
    output_df.loc['identity', :] = accumulate_errors(pred_tokens, orig_tokens, gold_tokens, vocab)

    # besttype
    pred_tokens = list(besttype_prediction(orig_tokens, gold_tokens))
    output_df.loc['besttype', :] = accumulate_errors(pred_tokens, orig_tokens, gold_tokens, vocab)

    # lexicon
    pred_tokens = list(lexicon_prediction(orig_tokens, train_lexicon))
    output_df.loc['lexicon', :] = accumulate_errors(pred_tokens, orig_tokens, gold_tokens, vocab)

    for filename in sys.argv[1:]:
        predicted_sentences = list(tqdm(read_file_sentences(filename, gold_sentences=gold_sentences), total=len(gold_sentences), leave=False))
        pred_tokens = list(itertools.chain.from_iterable(predicted_sentences))
        output_df.loc[Path(filename).name, :] = accumulate_errors(pred_tokens, orig_tokens, gold_tokens, vocab)

    output_df['overall'] = output_df.loc[slice(None), (slice(None), 'total')].sum(axis=1)

    print(output_df.to_string())
    print((output_df / output_df.loc['identity', ('overall', '')]).to_string(float_format=lambda x: f'{x*100:.3f}'))


if __name__ == '__main__':
    main()