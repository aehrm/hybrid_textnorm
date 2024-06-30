import more_itertools
import numpy as np
import pandas
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

from hybrid_textnorm.lexicon import Lexicon
from hybrid_textnorm.metrics import word_accuracy
from hybrid_textnorm.normalization import reranked_normalization, predict_type_normalization, prior_normalization


def main():
    # load the train dictionary
    train_lexicon = Lexicon.from_dataset('json', data_files='./dataset/processed/train.lexicon.jsonl', split='train')

    # load the dev dataset
    dev_dataset = load_dataset('json', data_files='./dataset/processed/dev.jsonl', split='train')\
                   .filter(lambda x: len(x['tokens']['orig']) < 200, batched=False)#.shuffle(seed=123).select(range(10000))

    dev_oov_types = set()
    for row in dev_dataset:
        dev_oov_types |= set(row['tokens']['orig']) - train_lexicon.keys()

    type_model_tokenizer = AutoTokenizer.from_pretrained('./model_output')
    type_model = AutoModelForSeq2SeqLM.from_pretrained('./model_output').cuda()

    oov_replacement_probabilities = {}
    with tqdm(total=len(dev_oov_types)) as pbar:
        for orig_type, normalizations in predict_type_normalization(dev_oov_types, type_model_tokenizer, type_model,
                                                                    batch_size=512):
            oov_replacement_probabilities[orig_type] = normalizations
            pbar.update()

    type_model.cpu()

    language_model_tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
    language_model = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2").cuda()
    language_model_tokenizer.add_special_tokens({'pad_token': '<pad>'})

    output_df = pandas.DataFrame()

    parameters = [(a, a) for a in [0, 0.2, 0.4, 0.6, 0.8, 1.0, 2, 3, 4]]
    for i, (alpha, beta) in enumerate(parameters):
        gold_tokens = []
        pred_tokens = []
        orig_tokens = []
        print(f'alpha={alpha}, beta={beta}')
        for row in tqdm(dev_dataset):
            if alpha == 0 and beta == 0:
                best_pred_tokens = prior_normalization(row['tokens']['orig'], train_lexicon, oov_replacement_probabilities)
            else:
                pred = reranked_normalization(row['tokens']['orig'], train_lexicon, oov_replacement_probabilities, language_model_tokenizer, language_model, alpha=alpha, beta=beta, batch_size=96)
                best_pred_tokens, _, _, _ = pred[0]

            gold_tokens.extend(row['tokens']['norm'])
            orig_tokens.extend(row['tokens']['orig'])
            pred_tokens.extend(best_pred_tokens)

        acc = word_accuracy(gold_tokens=gold_tokens, pred_tokens=pred_tokens, orig_tokens=orig_tokens, train_vocab_tokens=train_lexicon.keys())
        print(*acc)
        output_df.loc[i, 'alpha'] = alpha
        output_df.loc[i, 'beta'] = beta
        output_df.loc[i, 'overall'] = acc[0]
        output_df.loc[i, 'invocab'] = acc[1]
        output_df.loc[i, 'oov'] = acc[2]


    output_df.to_csv('model_output/rerank_search_results.csv')

if __name__ == '__main__':
    main()
