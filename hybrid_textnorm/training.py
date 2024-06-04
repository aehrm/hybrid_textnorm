import itertools

import collections
from datasets import Dataset
from tokenizers.decoders import BPEDecoder
from tqdm import tqdm

from transformers import PreTrainedTokenizerFast

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Replace, Sequence
from tokenizers.processors import RobertaProcessing
from tokenizers.tokenizers import Regex
from tokenizers.trainers import BpeTrainer

def train_tokenizer(lexicon, character_model=False, vocab_size=200, model_max_length=100):
    num_pairs = sum(sum(frequencies.values()) for frequencies in lexicon.values())
    def gen_types():
        for orig_token, norm_frequencies in lexicon.items():
            for norm_token, freq in norm_frequencies.items():
                for _ in range(freq):
                    assert ' ' not in orig_token
                    assert ' ' not in norm_token
                    yield orig_token
                    yield norm_token

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.decoder = BPEDecoder()

    if character_model:
        alphabet = list(set(itertools.chain.from_iterable(gen_types())))
        trainer = BpeTrainer(vocab_size=len(alphabet), initial_alphabet=alphabet, special_tokens=['<pad>', '<s>', '</s>', '<unk>'])
        tok.train_from_iterator([], trainer=trainer)
    else:
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=['<pad>', '<s>', '</s>', '<unk>'], show_progress=True)
        tok.train_from_iterator(gen_types(), trainer=trainer, length=num_pairs * 2)

    tok.enable_padding(pad_token='<pad>')
    tok.post_processor = RobertaProcessing(
        cls=("<s>", tok.token_to_id("<s>")),
        sep=("</s>", tok.token_to_id("</s>"))
    )

    # NOTE: this normalizer is not strictly necessary, but we add it to the tokenizer
    # such that an uninformed user omitting the pre-processing still gets the same result.
    tok.normalizer = Sequence([
        Replace(' ', ''),  # NOTE: Whitespace should not be a character in the type-based transformer
        Replace('ſ', 's'),
        Replace('a\u0364', 'ä'),
        Replace('o\u0364', 'ö'),
        Replace('u\u0364', 'ü')
    ])

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, model_max_length=model_max_length, pad_token='<pad>', unk_token='<unk>', bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>')
    return tokenizer
