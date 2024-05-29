import json

import itertools

import pandas
from datasets import DatasetDict, Dataset
from pathlib import Path
from tqdm import tqdm
import regex as re

from hybrid_textnorm.align_levenshtein import align_token_sequences
from hybrid_textnorm.preprocess import xml_to_samples, sequences_to_lexicon

from datasets import disable_caching
disable_caching()

DATASET_PATH = Path(__file__).parent / 'dataset'


def generate_samples(filenames):
    for filename in filenames:
        path = DATASET_PATH / 'dtaec-0.03' / 'xml' / filename
        yield from xml_to_samples(str(path))

def write_token_lines(lines, filename):
    with open(filename, 'w') as f:
        for line in lines:
            print(line, file=f)

    print(f'writing to {filename}')



def main():
    metadata_df = pandas.read_csv(Path(__file__).parent / 'dataset' / 'dataset_split.tsv', sep='\t')

    datasets_dict = {}
    for split, group in metadata_df.groupby('split'):
        dataset_list = list(tqdm(generate_samples(group['filename'].values), unit='examples', desc=split))
        datasets_dict[split] = Dataset.from_list(dataset_list)

    dataset = DatasetDict(datasets_dict)

    # main dataset file for us
    for split, split_dataset in dataset.items():
        path = DATASET_PATH / "processed" / f"{split}.jsonl"
        split_dataset.to_json(path)
        print(f'writing to {path}')

        lexicon = sequences_to_lexicon(split_dataset['tokens'])
        with open(DATASET_PATH / "processed" / f'{split}.lexicon.json', 'w') as f:
            json.dump(lexicon, f)
            print(f'writing lexicon to {f.name}')

    # dataset file for baseline systems
    for split, split_dataset in dataset.items():
        cognate_pairs = list(
            itertools.chain.from_iterable(zip(sent['orig'], sent['norm']) for sent in split_dataset['tokens']))
        write_token_lines([orig for orig, norm in cognate_pairs], DATASET_PATH / "processed" / f'{split}.orig')
        write_token_lines([norm for orig, norm in cognate_pairs], DATASET_PATH / "processed" / f'{split}.norm')
        write_token_lines([orig + '\t' + norm for orig, norm in cognate_pairs],
                          DATASET_PATH / "processed" / f'{split}.parallel')

        trans = str.maketrans("", "", '░▁')
        cognate_pairs_without_specials = [(orig, norm.translate(trans)) for orig, norm in cognate_pairs]
        write_token_lines([orig for orig, norm in cognate_pairs_without_specials],
                          DATASET_PATH / "processed" / f'{split}.nospacing.orig')
        write_token_lines([norm for orig, norm in cognate_pairs_without_specials],
                          DATASET_PATH / "processed" / f'{split}.nospacing.norm')
        write_token_lines([orig + '\t' + norm for orig, norm in cognate_pairs_without_specials],
                          DATASET_PATH / "processed" / f'{split}.nospacing.parallel')



if __name__ == '__main__':
    main()