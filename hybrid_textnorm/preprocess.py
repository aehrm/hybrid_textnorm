import json

import collections
from datasets import Dataset

from pathlib import Path

import regex as re
from lxml import etree
import more_itertools
from tqdm import tqdm

starts_punct = re.compile(r'^\p{Punct}')

def german_transliterate(s):
    return s.replace('ſ', 's') \
        .replace('a\u0364', 'ä') \
        .replace('o\u0364', 'ö') \
        .replace('u\u0364', 'ü') \
        .replace('…', '...')

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

def sequences_to_lexicon(training_aligned_sequences):
    num_pairs = sum(len(sentence['orig']) for sentence in training_aligned_sequences)
    lexicon = {}

    with tqdm(desc='Learning lexicon', unit='pairs', total=num_pairs) as pbar:
        for sentence in training_aligned_sequences:
            for orig_token, norm_token in zip(sentence['orig'], sentence['norm']):
                if orig_token not in lexicon.keys():
                    lexicon[orig_token] = collections.Counter()
                lexicon[orig_token].update([norm_token])
                pbar.update()

    return lexicon

def lexicon_to_translation_dataset(type_lexicon):
    def gen_dataset():
        for orig_token, norm_frequencies in type_lexicon.items():
            most_frequent_norm_token, _ = norm_frequencies.most_common(1)[0]
            yield {'orig': orig_token, 'norm': most_frequent_norm_token}

    return Dataset.from_generator(gen_dataset)

def load_lexicon(json_path):
    with open(json_path) as f:
        obj = json.load(f)

    return {orig_token: collections.Counter(v) for orig_token, v in obj.items()}