# Hybrid Model Text Normalization

A hybrid model to normalize historic text to contemporary orthography.

Huggingface repositories:
* [DTA Parallel Corpus Lexica](https://huggingface.co/datasets/aehrm/dtaec-lexica)
* [Type Normalization Transformer](https://huggingface.co/aehrm/dtaec-type-normalizer)

## Quickstart

1\. Clone and install the repository.

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

## API Usage

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
    --ref_file dataset/processed/test.jsonl \
    --orig_file dataset/processed/test.jsonl \  # optional if you want oov results
    --lexicon_file dataset/processed/train.lexicon.jsonl \
    --input_file model_output/*.pred
    # or --input_file baselines/output/*.pred model_output/*.pred if you wand to include the baselines
```
