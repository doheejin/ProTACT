# Prompt- and Trait Relation-aware Cross-prompt Essay Trait Scoring (ProTACT)

This repository is the implementation of the ProTACT architecture, introduced in the paper, [**Prompt- and Trait Relation-aware Cross-prompt Essay Trait Scoring**](https://aclanthology.org/2023.findings-acl.98/) (ACL Findings 2023).

> Our code is based on the open source, [https://github.com/robert1ridley/cross-prompt-trait-scoring](https://github.com/robert1ridley/cross-prompt-trait-scoring) (Ridley, 2021).

## Citation
Please cite our paper if you find this repository helpful.
```
@inproceedings{do-etal-2023-prompt,
    title = "Prompt- and Trait Relation-aware Cross-prompt Essay Trait Scoring",
    author = "Do, Heejin  and
      Kim, Yunsu  and
      Lee, Gary Geunbae",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.98",
    doi = "10.18653/v1/2023.findings-acl.98",
    pages = "1538--1551",
    abstract = "Automated essay scoring (AES) aims to score essays written for a given prompt, which defines the writing topic. Most existing AES systems assume to grade essays of the same prompt as used in training and assign only a holistic score. However, such settings conflict with real-education situations; pre-graded essays for a particular prompt are lacking, and detailed trait scores of sub-rubrics are required. Thus, predicting various trait scores of unseen-prompt essays (called cross-prompt essay trait scoring) is a remaining challenge of AES. In this paper, we propose a robust model: prompt- and trait relation-aware cross-prompt essay trait scorer. We encode prompt-aware essay representation by essay-prompt attention and utilizing the topic-coherence feature extracted by the topic-modeling mechanism without access to labeled data; therefore, our model considers the prompt adherence of an essay, even in a cross-prompt setting. To facilitate multi-trait scoring, we design trait-similarity loss that encapsulates the correlations of traits. Experiments prove the efficacy of our model, showing state-of-the-art results for all prompts and traits. Significant improvements in low-resource-prompt and inferior traits further indicate our model{'}s strength.",
}
```

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
