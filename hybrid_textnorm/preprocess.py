import logging

import io

import json

import collections
from datasets import Dataset

from pathlib import Path

import regex as re
from lxml import etree
import more_itertools
from tqdm import tqdm

from hybrid_textnorm.align_levenshtein import align_token_sequences
from hybrid_textnorm.detokenizer import DtaEvalDetokenizer

logger = logging.getLogger(__name__)

SPACE = re.compile(r'[ ▁]+')
MERGE_MARKS = re.compile(r'░+ *')

DETOKENIZER = DtaEvalDetokenizer()

def german_transliterate(s):
    return s.replace('ſ', 's') \
        .replace('a\u0364', 'ä') \
        .replace('o\u0364', 'ö') \
        .replace('u\u0364', 'ü') \
        .replace('…', '...')

def recombine_tokens(toks):
    text = ' '.join(toks)
    text = SPACE.sub(' ', text)
    text = MERGE_MARKS.sub('', text)
    return text.split(' ')


def xml_to_samples(filename):
    f = open(filename)
    tree = etree.parse(f)

    for sentence_id, sent in enumerate(tree.xpath('//s')):
        if sent.get('sbad', '0') == '1':
            continue

        tokens = sent.xpath('./w')
        if any(tok.get('bad', None) == '1' for tok in tokens):
            continue
        if any(tok.get('class', None) == 'BUG' for tok in tokens):
            continue
        if any(tok.get('class', None) == 'FM' for tok in tokens):
            continue
        if any(tok.get('class', None) == 'GRAPH' for tok in tokens):
            continue
        if any(tok.get('old', None) is None for tok in tokens):
            continue
        if any(tok.get('new', None) is None for tok in tokens):
            continue

        tokens_orig = []
        tokens_norm = []
        for tok in tokens:
            if tok.get('class', None) == 'JOIN':
                alignment = align_token_sequences(tok.get('old').split(' '), [tok.get('new')])
                for orig_subtok, norm_subtok in alignment:
                    tokens_orig.append(orig_subtok)
                    tokens_norm.append(norm_subtok)
            else:
                tokens_orig.append(tok.get('old'))
                tokens_norm.append(tok.get('new').replace(' ', '▁'))


        trans = {'orig': DETOKENIZER.detokenize(tokens_orig), 'norm': DETOKENIZER.detokenize(recombine_tokens(tokens_norm))}
        tokens = {'orig': list(map(german_transliterate, tokens_orig)), 'norm': list(map(german_transliterate, tokens_norm))}

        if not (
                all(re.fullmatch(r'[^ ▁░]+', tok) for tok in tokens['orig'])
                and all(re.fullmatch(r'[^ ]+░?', tok) for tok in tokens['norm'])
                and re.fullmatch(r'[^▁░]+', trans['orig'])
                and re.fullmatch(r'[^▁░]+', trans['norm'])
                and all(re.search(r'\p{Ll}\p{Lu}', tok) is None for tok in tokens['norm'])
        ):
            logger.warning(
                f'Something very bad happened while processing sentence #{sentence_id}; check source xml {filename}\n'
                f'Will skip this sentence.')
            continue

        assert all(re.fullmatch(r'[^ ▁░]+', tok) for tok in tokens['orig'])
        assert all(re.fullmatch(r'[^ ]+░?', tok) for tok in tokens['norm'])
        assert re.fullmatch(r'[^▁░]+', trans['orig'])
        assert re.fullmatch(r'[^▁░]+', trans['norm'])
        assert all(re.search(r'\p{Ll}\p{Lu}', tok) is None for tok in tokens['norm'])

        yield {'translation': trans, 'tokens': tokens, 'filename': Path(f.name).name, 'sentence_id': sentence_id}

    f.close()
