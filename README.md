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
| _Best theoret. type map_                                          |      99.547 |                99.533 |            99.896 |              22.612 |
| [Csmtiser](https://github.com/clarinsi/csmtiser) (sentence-level) |      98.928 |                99.317 |            89.160 |              21.151 |
| [Csmtiser](https://github.com/clarinsi/csmtiser) (token-level)    |      98.940 |                99.321 |            89.369 |              19.997 |
| [Norma](https://github.com/comphist/norma)                        |      96.834 |                99.477 |            30.521 |              23.392 |
| [Transnormer](https://github.com/ybracke/transnormer)             |      98.979 |                99.271 |            91.653 |              24.937 |
| hybrid_textnorm w/o LLM                                           |      99.111 |                99.481 |            89.823 |          **19.834** |
| hybrid_textnorm                                                   |  **99.196** |            **99.495** |        **91.701** |              20.451 |


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

Todo

## Reproduction

### Generating the dataset

```bash
cd dataset
wget 'https://kaskade.dwds.de/~moocow/software/dtaec/dtaec-0.03.tar.gz'
sha256sum -c dtaec-0.03.tar.gz.sha256sum
tar xvf dtaec-0.03.tar.gz
cd ..

poetry run python generate_dataset.py
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
