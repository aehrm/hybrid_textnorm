# Hybrid Model Text Normalization

A hybrid model to normalize historic text to contemporary orthography.

Huggingface repositories:
* [DTA Parallel Corpus Lexica](https://huggingface.co/datasets/aehrm/dtaec-lexica)
* [Type Normalization Transformer](https://huggingface.co/aehrm/dtaec-type-normalizer)

The project uses poetry for dependency management. You can just run `poetry install` to install all dependencies.

You may open a shell with `poetry shell` with all required python packages and interpreter. Alternatively, you can run scripts with the project-dependent python interpreter with `poetry run python <script.py>`.

## Quickstart

1\. Clone the repository and install the dependencies.

```bash
git clone https://github.com/aehrm/hybrid_textnorm
cd hybrid_textnorm
poetry install --no-root
```

2\. Start normalizing text.

```bash
echo "Im Nothfall könnte ich bey meinen gelehrten Freunden eine Anleihe von Citaten machen." > input_file
poetry run python normalize.py --input_file input_file
```

## Performance

Scores on a test set of the pre-transliterated [DTA-EC parallel corpus](https://kaskade.dwds.de/~moocow/software/dtaec). 
Contains 16 documents, ~36k sentences, ~701k tokens. Approximately 3.833% of tokens are out-of-vocabulary
(i.e. not seen in the training set).

|                                                                   | **WordAcc** | **WordAcc (invocab)** | **WordAcc (oov)** | **CER<sub>I</sub>** |
|:------------------------------------------------------------------|------------:|----------------------:|------------------:|--------------------:|
| _Identity_                                                        |      96.513 |                97.015 |            83.912 |              20.715 |
| _Lexicon_                                                         |      98.881 |                99.477 |            83.912 |          **18.767** |
| _Best theoret. type map_                                          |      99.547 |                99.533 |            99.896 |              22.612 |
| [Csmtiser](https://github.com/clarinsi/csmtiser) (sentence-level) |      98.928 |                99.317 |            89.160 |              21.151 |
| [Csmtiser](https://github.com/clarinsi/csmtiser) (token-level)    |      98.940 |                99.321 |            89.369 |              19.997 |
| [Norma](https://github.com/comphist/norma)                        |      96.834 |                99.477 |            30.521 |              23.392 |
| [Transnormer](https://github.com/ybracke/transnormer)             |      98.979 |                99.271 |            91.653 |              24.937 |
| hybrid_textnorm w/o LLM                                           |      99.111 |                99.481 |            89.823 |              19.834 |
| hybrid_textnorm                                                   |  **99.194** |            **99.493** |        **91.701** |              20.451 |


## Usage

```
usage: normalize.py [-h]
                    [--lexicon_dataset_name LEXICON_DATASET_NAME | --lexicon_file LEXICON_FILE | --no_lexicon]
                    [--type_model TYPE_MODEL | --no_type_model]
                    [--type_model_batch_size TYPE_MODEL_BATCH_SIZE]
                    [--language_model LANGUAGE_MODEL | --no_language_model]
                    [--language_model_batch_size LANGUAGE_MODEL_BATCH_SIZE] [--alpha ALPHA]
                    [--beta BETA] [--is_pretokenized] [--input_file INPUT_FILE]
                    [--output_file OUTPUT_FILE] [--output_text]

options:
  -h, --help            show this help message and exit
  --lexicon_dataset_name LEXICON_DATASET_NAME
                        Name of the dataset containing the lexicon (default: aehrm/dtaec-lexica)
  --lexicon_file LEXICON_FILE
                        JSON lexicon file
  --no_lexicon          Do not use lexicon for normalization
  --type_model TYPE_MODEL
                        Type model to be used (default: aehrm/dtaec-type-normalizer).
  --no_type_model       Do not use type model for normalization
  --type_model_batch_size TYPE_MODEL_BATCH_SIZE
                        Batch size for the type model (default: 64).
  --language_model LANGUAGE_MODEL
                        Language model to be used (default: dbmdz/german-gpt2)
  --no_language_model   Do not use language model for normalization
  --language_model_batch_size LANGUAGE_MODEL_BATCH_SIZE
                        Batch size for the language model (default: 8)
  --alpha ALPHA         Alpha parameter for model weighting (default: 0.5)
  --beta BETA           Beta parameter for model weighting (default: 0.5)
  --is_pretokenized     Supplied input is already whitespace-tokenized; skip tokenization
  --input_file INPUT_FILE
                        Input file path; use "-" for standard input (default: stdin)
  --output_file OUTPUT_FILE
                        Output file path; use "-" for standard output (default: stdout)
  --output_text         Output will be formatted as recombined detokenized text
```

## API

You can use the normalizer programmatically using the API. To install the normalizer in your project, use for instance
```bash
pip install git+https://github.com/aehrm/hybrid_textnorm
```

Then, you can start normalizing like this:
```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from hybrid_textnorm.lexicon import Lexicon
from hybrid_textnorm.normalization import predict_type_normalization, reranked_normalization, prior_normalization

lexicon_dataset_name = 'aehrm/dtaec-lexica'
type_model_name = 'aehrm/dtaec-type-normalizer'
language_model_name = 'dbmdz/german-gpt2'

train_lexicon = Lexicon.from_dataset(lexicon_dataset_name, split='train')
type_model_tokenizer = AutoTokenizer.from_pretrained(type_model_name)
type_model = AutoModelForSeq2SeqLM.from_pretrained(type_model_name)

hist_sentence = ['Wers', 'nicht', 'glaubt', ',', 'bezahlt', 'einen', 'Thaler', '.']

# generate normalization hypotheses for the oov types
oov_types = set(hist_sentence) - train_lexicon.keys()
if torch.cuda.is_available():
    type_model.cuda()

oov_replacement_probabilities = dict(predict_type_normalization(oov_types, type_model_tokenizer, type_model))
type_model.cpu()

# # if you want to skip the language model reranking:
# prior_pred = prior_normalization(hist_sentence, train_lexicon, oov_replacement_probabilities)
# print(prior_pred)

# rerank with the language model
language_model_tokenizer = AutoTokenizer.from_pretrained(language_model_name)
language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
if 'pad_token' not in language_model_tokenizer.special_tokens_map:
    language_model_tokenizer.add_special_tokens({'pad_token': '<pad>'})

if torch.cuda.is_available():
    language_model.cuda()

predictions = reranked_normalization(hist_sentence, train_lexicon, oov_replacement_probabilities, language_model_tokenizer, language_model)
best_pred, _, _, _ = predictions[0]
print(best_pred)
# >>> ['Wer▁es', 'nicht', 'glaubt', ',', 'bezahlt', 'einen', 'Taler', '.']
```

## Reproduction

### Generating the dataset

```bash
cd dataset
wget 'https://kaskade.dwds.de/~moocow/software/dtaec/dtaec-0.03.tar.gz'
sha256sum -c dtaec-0.03.tar.gz.sha256sum
tar xvf dtaec-0.03.tar.gz
cd ..

poetry run python prepare_dataset.py --write_baseline_format
```

### Training

```bash
poetry run python train.py --output_dir model_output
```

### Running the model on the test set

```bash
poetry run python normalize.py \
    --type_model model_output \
    --lexicon_file dataset/processed/train.lexicon.jsonl \
    --input_file dataset/processed/test.jsonl \
    --output_file model_output/test.pred
    
poetry run python normalize.py \
    --type_model model_output \
    --lexicon_file dataset/processed/train.lexicon.jsonl \
    --input_file dataset/processed/test.jsonl \
    --no_language_model \
    --output_file model_output/test.nolm.pred
```

### (Optional) Running the baseline systems for reproduction

```bash
docker build --tag ehrmanntraut/csmtiser_sentence baselines/csmtiser_sentence
docker build --tag ehrmanntraut/csmtiser_token baselines/csmtiser_token
docker build --tag ehrmanntraut/norma baselines/norma
docker build --tag ehrmanntraut/transnormer baselines/transnormer

# all of these may take a long time since the models are trained
docker run --rm -it -v $(pwd)/dataset/processed:/dataset -v $(pwd)/baselines/output:/output ehrmanntraut/norma
docker run --rm -it -v $(pwd)/dataset/processed:/dataset -v $(pwd)/baselines/output:/output ehrmanntraut/csmtiser_token
docker run --rm -it -v $(pwd)/dataset/processed:/dataset -v $(pwd)/baselines/output:/output ehrmanntraut/csmtiser_sentence
docker run --rm -it -v $(pwd)/dataset/processed:/dataset -v $(pwd)/baselines/output:/output ehrmanntraut/transnormer
# or with gpus: docker run --rm -it --gpus -v $(pwd)/dataset/processed:/dataset -v $(pwd)/baselines/output:/output ehrmanntraut/transnormer

poetry run python baselines/cab/fetch_cab_normalization.py
poetry run python baselines/cab/fetch_cab_normalization.py --disable-exlex
```

### Running the evaluation

```bash
poetry run python evaluate.py \
    --gold_file dataset/processed/test.jsonl \
    --orig_file dataset/processed/test.jsonl \  # optional if you want oov results
    --lexicon_file dataset/processed/train.lexicon.jsonl \
    --input_file model_output/*.pred
    # or --input_file baselines/output/*.pred model_output/*.pred if you wand to include the baselines
```
