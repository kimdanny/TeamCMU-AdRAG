# TeamCMU at Touché (CLEF 2025)

[![arXiv](https://img.shields.io/badge/arXiv-2507.00509-b31b1b.svg)](https://arxiv.org/abs/2507.00509)

Official code repository of TeamCMU's solution for  [Advertisement in RAG](https://touche.webis.de/clef25/touche25-web/advertisement-detection.html) shared task in Touché at [CLEF 2025](https://clef2025.clef-initiative.eu).


### Download the official shared task data: 

1. ```chmod +x ./download_data.sh```  
2. ```./download_data.sh```

## 1. QA System `qa_system/`

The QA System is responsible for generating contextually relevant responses to open-domain queries prior to any advertisement integration.


## 2. Synthetic Data Generation

To make a robust Ad Classifier, we attempt to train a classifier with different types of synthetic data.


### 2-1. Naive Synthetic Data Generation `synthetic_data_naive/`

NaiveSynthetic data generation follows the original Webis-Ads dataset approach, i.e., given an answer without an advertisement, prompt an LLM to inject an ad. The query generation prompts include no specific item; rather, the LLM is instructed to generate an advertisement of an item that fits the context, which may result in the creation of fictional products. To promote diversity, we use a combination of 5 different LLMs: GPT-4o, Gemma-2-9B-it, LLaMA-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and Mistral-7B-Instruct. Moreover, we devise 12 different prompts for ad insertion, targeting various advertising strategies (e.g., direct, indirect, explicit, implicit, hard-sell and soft-sell).

[Version 0.1](https://huggingface.co/jmvcoelho/ad-classifier-v0.1) and [Version 0.2](https://huggingface.co/jmvcoelho/ad-classifier-v0.2) Ad-Classifiers were trained with the combination of origial Webis-Ads dataset and Navie Synthetic dataset.

### 2-2. Structured Synthetic Data Generation `synthetic_data_structured/`

We attempt to create synthetic data with more structured way with non-ficational items. We create hard positive instances (indirect and implicit advertisements) and hard negative instances (neutral but informative responses).

**`01_collect_product_pages.py`**  
Collect product papges from Wikipedia, filtering them through Wikidata properties that indicate the page is about a product.
Collected pages are then sorted by the date of release.

**`02_summarize_text.py`**   
Fetch Wikipedia page of a product, and summarize product description focusing on the quality of the product to be advertised. Summarization prompt can be found in the file.

**`03_generate_synthetic_data.py`**  
Generate synthetic ads/non-ads dataset. 
Hard Positive instances are indirect and implicit advertisement, while
Hard Negative instances are factual and informative descriptions without any promotional intent.
Data generation prompts can be found in the file.


[Version 0.3](https://huggingface.co/teknology/ad-classifier-v0.3), [Version 0.4](https://huggingface.co/teknology/ad-classifier-v0.4), and [Version 0.5](https://huggingface.co/teknology/ad-classifier-v0.5) Ad-Classifiers were trained with the combination of origial Webis-Ads dataset, Navie Synthetic dataset and Structured Synthetic dataset.


## 3. Ad-Classifier `ad_classifier/`
The Ad-Classifier is formulated as a standard binary text classification task: given a query and its corresponding response, the model predicts whether the response contains an advertisement. To build increasingly robust classifiers, we incrementally expand the training data with progressively harder examples derived from multiple sources (Webis-Ads --> NaiveSynthetic --> StructuredSynthetic).

- Training scripts for:
    - Version 0.3:  
    `v3_train_classifier_with_mixed_synthetic.py`
    - Version 0.4:   
    `v4_train_classifier_curriculum_mixed_synthetic.py`
    - Version 0.5:   
    `v5_train_classifier_curriculum_mixed_synthetic_sampling.py`

- Test your classifier:  
    - `test_classifiers.py`

- Trained Classifiers available in HuggingFace 🤗:
    - [Version 0.0](https://huggingface.co/jmvcoelho/ad-classifier-v0.0)
    - [Version 0.1](https://huggingface.co/jmvcoelho/ad-classifier-v0.1)
    - [Version 0.2](https://huggingface.co/jmvcoelho/ad-classifier-v0.2)
    - [Version 0.3](https://huggingface.co/teknology/ad-classifier-v0.3)
    - [Version 0.4](https://huggingface.co/teknology/ad-classifier-v0.4)
    - [Version 0.5](https://huggingface.co/teknology/ad-classifier-v0.5)
    

## 4. Ad-Rewriter `ad_rewriter/`
The Ad-Rewriter module takes as input a query, an ad-free QA response, and a product to be advertised. These elements are combined into a prompt, which
conditions the rewriting process. The goal of the Ad-Rewriter is to produce a fluent, contextually relevant, and minimally intrusive ad-integrated version of the original response. We experiment with two methods:
- zero-shot rewriting, and
- supervised fine-tuning-based rewriting

### 4-1. Zero-Shot Rewriting
- perform zero-shot rewriting: `zero_shot_rewriting.py`
- evaluate the rewritten responses by  
`make_classifier_labels_on_rewritten_responses.py` and `eval_rewritten_responses.py`


### 4-2. SFT-Based Rewriting
- generate sft training data by `train_sft_rewriter/generate_sft_train_data.py`
- train your rewriter by `train_sft_rewriter/train.py`
- evaluate the rewritten responses with  
`make_classifier_labels_on_rewritten_responses.py` and `eval_rewritten_responses.py`

### Reference
**CEUR**
```
coming soon
```

**ArXiv**
```
@article{kim2025teamcmu,
  title={TeamCMU at Touch$\backslash$'e: Adversarial Co-Evolution for Advertisement Integration and Detection in Conversational Search},
  author={Kim, To Eun and Coelho, Jo{\~a}o and Onilude, Gbemileke and Singh, Jai},
  journal={arXiv preprint arXiv:2507.00509},
  year={2025}
}
```
