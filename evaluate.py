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
from hybrid_textnorm.metrics import word_accuracy, cerI

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(levelname)s - %(message)s",
)


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
                    logger.warning(f'Detected unaligned sequences on {filename}, will perform alignment')
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


def calc_metrics(gold_tokens, pred_tokens, train_vocab_tokens=None, orig_tokens=None):
    wordacc = word_accuracy(gold_tokens=gold_tokens,
                            pred_tokens=pred_tokens,
                            train_vocab_tokens=train_vocab_tokens,
                            orig_tokens=orig_tokens)
    ceri = cerI(gold_tokens=gold_tokens,
                            pred_tokens=pred_tokens,
                            train_vocab_tokens=train_vocab_tokens,
                            orig_tokens=orig_tokens)

    return [wordacc['overall'], wordacc['invocab'], wordacc['oov'], ceri['overall']]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gold_file', type=str, required=True,
                        help='JSON file against which to compare the system output')
    parser.add_argument('--orig_file', type=str,
                        help='JSON file with the original input. Will be used to compute OOV scores')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--lexicon_dataset_name', type=str, default='aehrm/dtaec-lexica',
                        help='Name of the dataset containing the lexicon (default: %(default)s)')
    group.add_argument('--lexicon_file', type=str, help='Path to the lexicon JSON file.')
    parser.add_argument('--ignore_punct', default=True, action=argparse.BooleanOptionalAction,
                        help='Ignore punctuation during evaluation (default: %(default)s)')
    parser.add_argument('--ignore_case', default=False, action=argparse.BooleanOptionalAction,
                        help='Ignore casing during evaluation (default: %(default)s)')
    parser.add_argument('--ignore_space', default=False, action=argparse.BooleanOptionalAction,
                        help='Ignore whitespaces during evaluation (default: %(default)s)')
    parser.add_argument('--align', default='auto', const='auto', nargs='?', choices=['auto', 'never', 'always'],
                        help="Specifies how text alignment should be handled. Options: 'auto', 'never', or 'always' (default: %(default)s)")
    parser.add_argument('--output-csv', action='store_true',
                        help='Output the evaluation results in CSV format')
    parser.add_argument('--input_file', type=str, required=True, nargs='+',
                        help='Path to one or more input files to be evaluated')

    args = parser.parse_args()
    logger.setLevel(logging.INFO)

    trans = str.maketrans("", "", '░▁')
    is_punct = re.compile(r'\p{Punct}+')
    def map_tokens(tokens, gold_tokens):
        for token, gold in zip(tokens, gold_tokens):
            if args.ignore_punct and is_punct.fullmatch(gold):
                # if we ignore punctuation, we assume that the system makes a perfect prediction
                yield gold
            else:
                if args.ignore_case:
                    token = token.lower()
                if args.ignore_space:
                    token = token.translate(trans)
                yield token

    logger.info('loading gold file')
    gold_sentences = list(read_file_sentences(args.gold_file, dataset_token_field='norm'))

    train_vocab = None
    orig_sentences = None
    columns = ['word_acc', 'cerI']
    if (args.lexicon_dataset_name or args.lexicon_file) and args.orig_file:
        logger.info('loading train lexicon')
        if args.lexicon_file:
            train_vocab = Lexicon.from_dataset('json', data_files=args.lexicon_file, split='train').keys()
        elif args.lexicon_dataset_name:
            train_vocab = Lexicon.from_dataset(args.lexicon_dataset_name, split='train').keys()

        logger.info('loading orig file')
        orig_sentences = list(read_file_sentences(args.orig_file, dataset_token_field='orig'))
        columns = ['word_acc', 'word_acc_invocab', 'word_acc_oov', 'cerI']

    gold_tokens = list(itertools.chain.from_iterable(gold_sentences))
    orig_tokens = list(itertools.chain.from_iterable(orig_sentences)) if orig_sentences is not None else None

    # pre-process tokens to fit the desired evaluation type
    gold_tokens = list(map_tokens(gold_tokens, gold_tokens))

    output_df = pandas.DataFrame(columns=columns)
    if orig_tokens:
        logger.info('evaluating identity')
        identity_metrics = calc_metrics(gold_tokens=gold_tokens,
                                         pred_tokens=orig_tokens,
                                         train_vocab_tokens=train_vocab,
                                         orig_tokens=orig_tokens)
        output_df.loc['identity'] = identity_metrics

        logger.info('evaluating best theoret. type')
        besttype_metrics = calc_metrics(gold_tokens=gold_tokens,
                                         pred_tokens=besttype_prediction(orig_tokens, gold_tokens),
                                         train_vocab_tokens=train_vocab,
                                         orig_tokens=orig_tokens)
        output_df.loc['besttype'] = besttype_metrics

    for filename in args.input_file:
        if filename.endswith('README.md'):
            # this surely is not an evaluation file ...
            logger.info(f'skipping {filename}')
            continue

        logger.info(f'evaluating {filename}')
        predicted_sentences = list(tqdm(read_file_sentences(filename, align=args.align, gold_sentences=gold_sentences), total=len(gold_sentences), leave=False))
        pred_tokens = list(itertools.chain.from_iterable(predicted_sentences))
        # pre-process tokens to fit the desired evaluation type
        pred_tokens = list(map_tokens(pred_tokens, gold_tokens))
        metrics = calc_metrics(gold_tokens=gold_tokens,
                                pred_tokens=pred_tokens,
                                train_vocab_tokens=train_vocab,
                                orig_tokens=orig_tokens)

        output_df.loc[Path(filename).name] = metrics

    if args.output_csv:
        print(output_df.to_csv())
    else:
        print(output_df.to_string())

if __name__ == '__main__':
    main()
