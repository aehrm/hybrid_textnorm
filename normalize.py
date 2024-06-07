
import sys

import logging

import argparse
import torch
from datasets import load_dataset

from somajo import SoMaJo
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

import regex as re

from hybrid_textnorm.beam_search import make_tokens_to_llm_string
from hybrid_textnorm.lexicon import Lexicon
from hybrid_textnorm.normalization import predict_type_normalization, reranked_normalization
from hybrid_textnorm.preprocess import german_transliterate, recombine_tokens, tokens_to_string

logger = logging.getLogger(__name__)

def load_input(input_file, do_tokenize=False):
    if input_file.endswith('.jsonl'):
        ds = load_dataset('json', data_files=input_file, split='train')
        for row in ds:
            yield row['tokens']['orig']
    else:
        if input_file == '-':
            input_str = sys.stdin.read()
        else:
            with open(input_file, 'r') as f:
                input_str = f.read()

        if do_tokenize:
            logger.info('tokenizing/sentencizing')
            text_tokenizer = SoMaJo("de_CMC", split_camel_case=True)
            sentences = text_tokenizer.tokenize_text([input_str])
            for sentence in sentences:
                yield [german_transliterate(tok.text) for tok in sentence]
        else:
            for line in input_str.splitlines():
                if line != '':
                    yield [german_transliterate(tok) for tok in line.split(' ')]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lexicon_dataset_name', type=str, default='aehrm/dtaec-lexica')
    parser.add_argument('--lexicon_file', type=str)
    parser.add_argument('--no_lexicon', action='store_true')
    parser.add_argument('--no_type_model', action='store_true')
    parser.add_argument('--no_language_model', action='store_true')
    parser.add_argument('--do_tokenize', action='store_true')
    parser.add_argument('--type_model', type=str, default='aehrm/dtaec-type-normalizer')
    parser.add_argument('--type_model_batch_size', type=int, default=64)
    parser.add_argument('--language_model', type=str, default='dbmdz/german-gpt2')
    parser.add_argument('--language_model_batch_size', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.55)
    parser.add_argument('--beta', type=float, default=1.1)
    parser.add_argument('--input_file', type=str, required=True, default='-')
    parser.add_argument('--output_file', type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('--output_text', action='store_true')

    args = parser.parse_args()
    logging.basicConfig(
        format="%(levelname)s - %(message)s",
    )
    logger.setLevel(logging.INFO)

    train_lexicon = Lexicon()
    if not args.no_lexicon:
        logger.info('loading lexicon')
        if args.lexicon_dataset_name:
            train_lexicon = Lexicon.from_dataset(args.lexicon_dataset_name, split='train')
        elif args.lexicon_file:
            train_lexicon = Lexicon.from_dataset('json', data_files=args.lexicon_file, split='train')
        else:
            logger.error('no lexicon specified!')
            sys.exit(1)

    type_model_tokenizer = None
    type_model = None
    if not args.no_type_model:
        logger.info('loading type model')
        type_model_tokenizer = AutoTokenizer.from_pretrained(args.type_model)
        type_model = AutoModelForSeq2SeqLM.from_pretrained(args.type_model)

    language_model_tokenizer = None
    language_model = None
    if not args.no_type_model:
        logger.info('loading large language model')
        language_model_tokenizer = AutoTokenizer.from_pretrained(args.language_model)
        language_model = AutoModelForCausalLM.from_pretrained(args.language_model)
        if 'pad_token' not in language_model_tokenizer.special_tokens_map:
            language_model_tokenizer.add_special_tokens({'pad_token': '<pad>'})

    logger.info('collect input material')
    input_dataset = list(load_input(args.input_file, do_tokenize=args.do_tokenize))

    oov_types = set()
    for orig_sent in input_dataset:
        oov_types |= set(orig_sent) - train_lexicon.keys()

    oov_replacement_probabilities = {}
    if not args.no_type_model:
        logger.info('normalize oov types')

        if torch.cuda.is_available():
            type_model.cuda()
        with tqdm(total=len(oov_types)) as pbar:
            for orig_type, normalizations in predict_type_normalization(oov_types, type_model_tokenizer, type_model, batch_size=args.type_model_batch_size):
                oov_replacement_probabilities[orig_type] = normalizations
                pbar.update()

        type_model.cpu()
    else:
        oov_replacement_probabilities = {orig_type: [(orig_type, 1)] for orig_type in oov_types}

    def print_result(tokens):
        if args.output_text:
            print(tokens_to_string(recombine_tokens(tokens)), file=args.output_file)
        else:
            print(' '.join(tokens), file=args.output_file)


    do_rerank = True
    if args.no_language_model:
        do_rerank = False
    if args.alpha == 0 and args.beta == 0:
        logger.warning('parameters alpha and beta set to 0, thus language model reranking has no effect; will not run the language model')
        do_rerank = False

    if not do_rerank:
        logger.info('return the maximum prior normalization without language model reranking')
        for orig_sent in tqdm(input_dataset):
            pred = []
            for orig_tok in orig_sent:
                if orig_tok in train_lexicon.keys():
                    pred.append(train_lexicon[orig_tok].most_common(1)[0][0])
                else:
                    replacements = oov_replacement_probabilities[orig_tok]
                    best_replace, _ = max(replacements, key=lambda x: x[1])
                    pred.append(best_replace)

            print_result(pred)
    else:
        logger.info('reranking normalization hypotheses with the language model')
        if torch.cuda.is_available():
            language_model.cuda()
        for orig_sent in tqdm(input_dataset):
            if len(language_model_tokenizer(make_tokens_to_llm_string(orig_sent))['input_ids']) + 50 > language_model_tokenizer.model_max_length:
                logger.info('sentence too long, giving up')
                print_result(orig_sent)
                continue

            predictions = reranked_normalization(orig_sent, train_lexicon, oov_replacement_probabilities, language_model_tokenizer, language_model, alpha=args.alpha, beta=args.beta, batch_size=args.language_model_batch_size)
            best_pred, _, _, _ = predictions[0]
            print_result(best_pred)


if __name__ == '__main__':
    main()
