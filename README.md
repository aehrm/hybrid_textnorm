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
