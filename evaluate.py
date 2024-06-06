import pandas
from pathlib import Path

import regex as re

import logging

import argparse
from datasets import load_dataset

from hybrid_textnorm.lexicon import Lexicon
from hybrid_textnorm.metrics import word_accuracy

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(levelname)s - %(message)s",
)


def read_file_tokens(filename, dataset_token_field=None):
    if filename.endswith('.jsonl'):
        ds = load_dataset('json', data_files=filename, split='train')
        for row in ds:
            yield from row['tokens'][dataset_token_field]
    else:
        with open(filename) as f:
            lines = f.readlines()

        if all(' ' not in l for l in lines):
            # assume one token per line
            for line in lines:
                yield line.strip()
        else:
            # assume one (space-tokenized) sentence per line
            for line in lines:
                yield from line.strip().split(' ')


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
    ref_tokens = list(read_file_tokens(args.ref_file, dataset_token_field='norm'))

    train_vocab = None
    orig_tokens = None
    columns = ['word_acc']
    if (args.lexicon_dataset_name or args.lexicon_file) and args.orig_file:
        logger.info('loading train lexicon')
        if args.lexicon_file:
            train_vocab = Lexicon.from_dataset('json', data_files=args.lexicon_file, split='train').keys()
        elif args.lexicon_dataset_name:
            train_vocab = Lexicon.from_dataset(args.lexicon_dataset_name, split='train').keys()

        logger.info('loading orig file')
        orig_tokens = list(read_file_tokens(args.orig_file, dataset_token_field='orig'))
        columns = ['word_acc', 'word_acc_invocab', 'word_acc_oov']

    ljust = max(len(Path(x).name) for x in args.input_file) + 4
    print(''.ljust(ljust), *[c.rjust(18) for c in columns])
    for filename in args.input_file:
        predicted_tokens = list(read_file_tokens(filename))
        metrics = word_accuracy(gold_tokens=ref_tokens, pred_tokens=predicted_tokens, train_vocab_tokens=train_vocab, orig_tokens=orig_tokens, equality=equality)

        print(f'{Path(filename).name.ljust(ljust)}', *[f"{m: 18.5f}" for m in metrics])


if __name__ == '__main__':
    main()
