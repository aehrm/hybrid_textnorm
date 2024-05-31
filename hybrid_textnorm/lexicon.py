import collections.abc
from datasets import load_dataset, Dataset
from tqdm import tqdm


class Lexicon(collections.abc.Mapping):

    def __init__(self, lexicon_dict=None):
        if lexicon_dict is None:
            self.lexicon_dict = dict()
        else:
            self.lexicon_dict = lexicon_dict

    def __getitem__(self, __key):
        return self.lexicon_dict[__key]

    def __len__(self):
        return len(self.lexicon_dict)

    def __iter__(self):
        return iter(self.lexicon_dict)

    @classmethod
    def from_dataset(cls, dataset_or_name, **load_dataset_kwargs):
        if isinstance(dataset_or_name, str):
            dataset = load_dataset(dataset_or_name, **load_dataset_kwargs)
        else:
            dataset = dataset_or_name

        lexicon_dict = dict()
        for row in dataset:
            orig_token = row['orig']
            norm_token = row['norm']
            freq = row['freq']
            if orig_token not in lexicon_dict.keys():
                lexicon_dict[orig_token] = collections.Counter()
            lexicon_dict[orig_token][norm_token] = freq

        return cls(lexicon_dict)

    def to_dataset(self, k_most_common=None):
        def gen_dataset():
            for orig_token, norm_frequencies in sorted(self.items(), key=lambda x: x[0]):
                for norm_token, freq in norm_frequencies.most_common(k_most_common):
                    yield {'orig': orig_token, 'norm': norm_token, 'freq': freq}

        return Dataset.from_generator(gen_dataset)

    @classmethod
    def sequences_to_lexicon(cls, training_aligned_sequences):
        num_pairs = sum(len(sentence['orig']) for sentence in training_aligned_sequences)
        lexicon_dict = {}

        with tqdm(desc='Learning lexicon', unit='pairs', total=num_pairs) as pbar:
            for sentence in training_aligned_sequences:
                for orig_token, norm_token in zip(sentence['orig'], sentence['norm']):
                    if orig_token not in lexicon_dict.keys():
                        lexicon_dict[orig_token] = collections.Counter()
                    lexicon_dict[orig_token].update([norm_token])
                    pbar.update()

        return cls(lexicon_dict)

def load_lexicon(json_path):
    if isinstance(json_path, io.TextIOBase):
        obj = json.load(json_path)
    else:
        with open(json_path) as f:
            obj = json.load(f)

    return {orig_token: collections.Counter(v) for orig_token, v in obj.items()}
