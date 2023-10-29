# Prompt- and Trait Relation-aware Cross-prompt Essay Trait Scoring (ProTACT)

This repository is the implementation of the ProTACT architecture, introduced in the paper, [**Prompt- and Trait Relation-aware Cross-prompt Essay Trait Scoring**](https://aclanthology.org/2023.findings-acl.98/) (ACL Findings 2023).

> Our code is based on the open source, [https://github.com/robert1ridley/cross-prompt-trait-scoring](https://github.com/robert1ridley/cross-prompt-trait-scoring) (Ridley, 2021).

## Package Requirements

Install below packages in your virtual environment before running the code.
- python==3.7.11
- tensorflow=2.0.0
- numpy=1.18.1
- nltk=3.4.5
- pandas=1.0.5
- scikit-learn=0.22.1

## Download GloVe

For prompt word embedding, we use the pretrained GloVe embedding.
- Go to [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) and download `glove.6B.50d.txt`.
- Put downloaded file in the `embeddings` directory.

## Run ProTACT (Do, 2023)
This bash script will run each model 5 times with different seeds ([12, 22, 32, 42, 52]).
- `bash ./train_ProTACT.sh`

Note that every run does not produce the same results due to the random elements.

## Run baseline (Ridley, 2021)
This bash script will run each model 5 times with different seeds ([12, 22, 32, 42, 52]).
- `bash ./train_CTS.sh`