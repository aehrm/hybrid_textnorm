import itertools

import pandas
from pathlib import Path
from tqdm import tqdm

import regex as re

import logging

import argparse
from datasets import load_dataset, Dataset

from hybrid_textnorm.align_levenshtein import align_token_sequences
from hybrid_textnorm.lexicon import Lexicon
from hybrid_textnorm.metrics import word_accuracy

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(levelname)s - %(message)s",
)


def align_pred_sent_with_ref_sent(predicted_sent, ref_sent):
    original_spacing_tokens = []
    trans = str.maketrans("", "", '░▁')
    for ref_tok in ref_sent:
        original_spacing_tokens.append(ref_tok.translate(trans))

    alignment = align_token_sequences(original_spacing_tokens, predicted_sent)

    new_pred_sent = []
    for i in range(len(alignment)):
        if i < len(alignment) - 1 and re.match(r'\p{Punct}', alignment[i + 1][0]):
            new_pred_sent.append(alignment[i][1].replace('░', ''))
        else:
            new_pred_sent.append(alignment[i][1])

    return new_pred_sent


def read_file_sentences(filename, dataset_token_field=None, align='auto', ref_sentences=None):
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
                    logger.warning(f'Detected unaligned sequences on {filename}, will perform alignment')
                    do_align = True

            if do_align:
                assert len(lines) == len(ref_sentences)
                ds = []
                for line, ref_sent in zip(lines, ref_sentences):
                    predicted_sent = line.strip().split(' ')
                    ds.append({'predicted_sent': predicted_sent, 'ref_sent': ref_sent})

                ds = Dataset.from_list(ds)
                ds = ds.map(lambda x: {'aligned': align_pred_sent_with_ref_sent(x['predicted_sent'], x['ref_sent'])},
                            num_proc=4)

                for row in ds:
                    aligned = row['aligned']
                    ref_sent = row['ref_sent']
                    assert len(aligned) == len(ref_sent)
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



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ref_file', type=str, required=True)
    parser.add_argument('--orig_file', type=str)
    parser.add_argument('--lexicon_dataset_name', type=str, default='aehrm/dtaec-lexica')
    parser.add_argument('--lexicon_file', type=str)
    parser.add_argument('--input_file', type=str, required=True, nargs='+')
    parser.add_argument('--ignore_punct', default=True)
    parser.add_argument('--ignore_case', default=False)
    parser.add_argument('--ignore_space', default=False)
    parser.add_argument('--align', default='auto', const='auto', nargs='?', choices=['auto', 'never', 'always'])

    args = parser.parse_args()
    logger.setLevel(logging.INFO)

    trans = str.maketrans("", "", '░▁')
    is_punct = re.compile(r'\p{Punct}+')
    def equality(gold, pred):
        if args.ignore_punct and is_punct.fullmatch(gold):
            return True
        if args.ignore_case:
            gold = gold.lower()
            pred = pred.lower()
        if args.ignore_space:
            gold = gold.translate(trans)
            pred = pred.translate(trans)
        return gold == pred
            
        

    logger.info('loading ref file')
    ref_sentences = list(read_file_sentences(args.ref_file, dataset_token_field='norm'))

    train_vocab = None
    orig_sentences = None
    columns = ['word_acc']
    if (args.lexicon_dataset_name or args.lexicon_file) and args.orig_file:
        logger.info('loading train lexicon')
        if args.lexicon_file:
            train_vocab = Lexicon.from_dataset('json', data_files=args.lexicon_file, split='train').keys()
        elif args.lexicon_dataset_name:
            train_vocab = Lexicon.from_dataset(args.lexicon_dataset_name, split='train').keys()

        logger.info('loading orig file')
        orig_sentences = list(read_file_sentences(args.orig_file, dataset_token_field='orig'))
        columns = ['word_acc', 'word_acc_invocab', 'word_acc_oov']

    ljust = max(len(Path(x).name) for x in args.input_file) + 4
    print(''.ljust(ljust), *[c.rjust(18) for c in columns])

    ref_tokens = list(itertools.chain.from_iterable(ref_sentences))
    orig_tokens = list(itertools.chain.from_iterable(orig_sentences)) if orig_sentences is not None else None

    if orig_tokens:
        identity_metrics = word_accuracy(gold_tokens=ref_tokens,
                                         pred_tokens=orig_tokens,
                                         train_vocab_tokens=train_vocab,
                                         orig_tokens=orig_tokens,
                                         equality=equality)
        print('identity'.ljust(ljust), *[f"{m: 18.5f}" for m in identity_metrics])

        besttype_metrics = word_accuracy(gold_tokens=ref_tokens,
                                         pred_tokens=besttype_prediction(orig_tokens, ref_tokens),
                                         train_vocab_tokens=train_vocab,
                                         orig_tokens=orig_tokens,
                                         equality=equality)
        print('besttype'.ljust(ljust), *[f"{m: 18.5f}" for m in besttype_metrics])

    for filename in args.input_file:
        predicted_sentences = list(tqdm(read_file_sentences(filename, align=args.align, ref_sentences=ref_sentences), total=len(ref_sentences), leave=False))
        pred_tokens = list(itertools.chain.from_iterable(predicted_sentences))
        metrics = word_accuracy(gold_tokens=ref_tokens,
                                pred_tokens=pred_tokens,
                                train_vocab_tokens=train_vocab,
                                orig_tokens=orig_tokens,
                                equality=equality)

        print(f'{Path(filename).name.ljust(ljust)}', *[f"{m: 18.5f}" for m in metrics])


if __name__ == '__main__':
    main()
