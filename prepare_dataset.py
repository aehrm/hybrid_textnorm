import argparse

import json

import itertools

import pandas
from datasets import DatasetDict, Dataset
from pathlib import Path
from tqdm import tqdm

from datasets import disable_caching

from hybrid_textnorm.lexicon import Lexicon
from hybrid_textnorm.preprocess import xml_to_samples

disable_caching()

def generate_samples(filenames):
    for filename in filenames:
        yield from xml_to_samples(filename)

def write_token_lines(lines, filename):
    with open(filename, 'w') as f:
        for line in lines:
            print(line, file=f)

    print(f'writing to {filename}')



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_split', type=argparse.FileType('r'),
                        default=Path(__file__).parent / 'dataset' / 'dataset_split.tsv',
                        help='Path to split metadata table (default: %(default)s)')
    parser.add_argument('--dataset_path', type=Path,
                        default=Path(__file__).parent / 'dataset' / 'dtaec-0.03' / 'xml',
                        help='Path to DTAEC xml files (default: %(default)s)')
    parser.add_argument('--output_path', type=Path,
                        default=Path(__file__).parent / 'dataset' / 'processed',
                        help='Path to DTAEC xml files (default: %(default)s)')

    args = parser.parse_args()

    metadata_df = pandas.read_csv(args.dataset_split, sep='\t')

    datasets_dict = {}
    for split, group in metadata_df.groupby('split'):
        filenames = [args.dataset_path / name for name in group['filename'].values]
        dataset_list = list(tqdm(generate_samples(filenames), unit='examples', desc=split))
        datasets_dict[split] = Dataset.from_list(dataset_list)

    dataset = DatasetDict(datasets_dict)

    # main dataset file for us
    for split, split_dataset in dataset.items():
        path = args.output_path / f"{split}.jsonl"
        split_dataset.to_json(path)
        print(f'writing to {path}')

        lexicon_path = args.output_path / f'{split}.lexicon.jsonl'
        lexicon = Lexicon.sequences_to_lexicon(split_dataset['tokens'])
        lexicon_ds = lexicon.to_dataset()
        lexicon_ds.to_json(lexicon_path)
        print(f'writing to {lexicon_path}')

    print('writing auxiliary files for baseline systems')

    # dataset file for baseline systems
    for split, split_dataset in dataset.items():
        cognate_pairs = list(
            itertools.chain.from_iterable(zip(sent['orig'], sent['norm']) for sent in split_dataset['tokens']))
        write_token_lines([orig for orig, norm in cognate_pairs], args.output_path / f'{split}.orig')
        write_token_lines([norm for orig, norm in cognate_pairs], args.output_path / f'{split}.norm')
        write_token_lines([orig + '\t' + norm for orig, norm in cognate_pairs], args.output_path / f'{split}.parallel')

        trans = str.maketrans("", "", '░▁')
        cognate_pairs_without_specials = [(orig, norm.translate(trans)) for orig, norm in cognate_pairs]
        write_token_lines([orig for orig, norm in cognate_pairs_without_specials],
                          args.output_path / f'{split}.nospacing.orig')
        write_token_lines([norm for orig, norm in cognate_pairs_without_specials],
                          args.output_path / f'{split}.nospacing.norm')
        write_token_lines([orig + '\t' + norm for orig, norm in cognate_pairs_without_specials],
                          args.output_path / f'{split}.nospacing.parallel')



if __name__ == '__main__':
    main()