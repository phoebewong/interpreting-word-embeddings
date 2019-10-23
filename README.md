Interpreting High-Dimensional Word Embeddings
==============================
By Phoebe Wong and Christine Jou

Harvard CS282BR, Fall 2019

# Baseline for SPINE
Adopted from [SPINE: SParse Interpretable Neural Embeddings](https://arxiv.org/pdf/1711.08792.pdf) and [source code](https://github.com/harsh19/SPINE) by Subramanian et. al on generating interpretable word embeddings using a k-sparse autoencoder.

## Word Embedding Format

The input embeddings, that you wish to transform, should be in the following format. Each line contains the word and its continuous representation in a space separated format

```
word1 0.4 0.2 0.42 ...
word2 0.23 0.54 0.123 ...
```

SPINE word vectors of original glove and word2vec vectors, along with word vectors from the paper's baseline [Sparse Overcomplete Word Vectors](https://arxiv.org/abs/1506.02004) (SPOWV) method, are available [here](https://drive.google.com/open?id=1aBXhAqJ3eNjV9WwgC0sUCDr771e2w4I9). We have also already included these embeddings in `data/external/`.

## Generating SPINE Embeddings
To generate the SPINE embeddings from the original embeddings, we run the following command:

```
python src/models/main.py \
        --input input_file \
        --num_epochs 4000 \
        --denoising \
        --noise 0.2 \
        --sparsity 0.85 \
        --hdim 1000
```

Generated embeddings will be outputted in  `data/interim` as a file named `input_file + '.spine'` in the same word embedding format as described above. Helper classes and code are in `src/models/utils.py` and `src/models/model.py`. Default of paper's embeddings is a hidden dimension of 1000. Hyperparameters such as noise, number of epochs, batch size, sparsity, and denoising can be adjusted as described by option flags.

An working example would be:

```
python src/models/main.py --input data/external/word2vec_original_15k_300d_train.txt
```
To run the code, you need to download the data and store it in `data/external`

The data is available [here](https://drive.google.com/drive/folders/1ksVcWDADmnp0Cl5kezjHqTg3Jnh8q031?usp=sharing), provided by the original authors.

We use the SPINE embeddings output to do our analysis which we have included in `notebooks/` and in HW2 writeup.

## T-SNE and UMAP
In progress

## PCA + Kmeans
In progress


--------

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
