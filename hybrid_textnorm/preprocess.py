import io

import json

import collections
from datasets import Dataset

from pathlib import Path

import regex as re
from lxml import etree
import more_itertools
from tqdm import tqdm

from hybrid_textnorm.beam_search import SPACE, MERGE_MARKS

starts_punct = re.compile(r'^\p{Punct}')

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

def tokens_to_string(toks):
    toks = [' ' + tok if not starts_punct.match(tok) else tok for tok in toks]

    return ''.join(toks).strip()

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
            if tok.get('class', None) == 'SPLIT':
                tokens_orig.append(tok.get('old'))
                tokens_norm.append(tok.get('new').replace(' ', '▁'))
            elif tok.get('class', None) == 'JOIN':
                for _, is_last, subtok in more_itertools.mark_ends(tok):
                    tokens_orig.append(subtok.get('old'))
                    if not is_last:
                        tokens_norm.append(subtok.get('new') + '░')
                    else:
                        tokens_norm.append(subtok.get('new'))
            else:
                tokens_orig.append(tok.get('old'))
                tokens_norm.append(tok.get('new'))

        tokens_orig = list(map(german_transliterate, tokens_orig))
        tokens_norm = list(map(german_transliterate, tokens_norm))

        trans = {'orig': tokens_to_string(tokens_orig), 'norm': tokens_to_string(tokens_norm)}
        tokens = {'orig': tokens_orig, 'norm': tokens_norm}

        yield {'translation': trans, 'tokens': tokens, 'filename': Path(f.name).name, 'sentence_id': sentence_id}

    f.close()