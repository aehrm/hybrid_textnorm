# Hybrid Model Text Normalization

## Generating the dataset

```
cd dataset
wget 'https://kaskade.dwds.de/~moocow/software/dtaec/dtaec-0.03.tar.gz'
sha256sum -c dtaec-0.03.tar.gz.sha256sum
tar xvf dtaec-0.03.tar.gz
cd ..

poetry run generate_dataset.py
```

## Running the baseline systems for reproduction

```
docker build --tag ehrmanntraut/csmtiser_sentence baselines/csmtiser_sentence
docker build --tag ehrmanntraut/csmtiser_token baselines/csmtiser_token
docker build --tag ehrmanntraut/norma baselines/norma

docker run --rm -it -v $(pwd)/dataset/processed:/dataset -v $(pwd)/baselines/output:/output ehrmanntraut/norma
docker run --rm -it -v $(pwd)/dataset/processed:/dataset -v $(pwd)/baselines/output:/output ehrmanntraut/csmtiser_token
docker run --rm -it -v $(pwd)/dataset/processed:/dataset -v $(pwd)/baselines/output:/output ehrmanntraut/csmtiser_sentence

poetry run python baselines/cab/fetch_cab_normalization.py
```
