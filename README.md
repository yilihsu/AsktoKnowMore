# AsktoKnowMore
This repository contains code and models for the paper: "Ask to Know More: Counterfactual Explanations for Fake Claims"

## **General Framework**

![](https://i.imgur.com/wslKi1G.png)

## **Example of generated counterfactual explanations**

![](https://i.imgur.com/Ic8lvp4.png)

### **Installation**

#### QA generator

Download and extract zip of Sense2vec wordvectors that are used for generation of multiple choices.
```
wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
tar -xvf  s2v_reddit_2015_md.tar.gz
```
#### QA-to-Claim Model
Download the pretrained QA2D model from the Google Cloud here. You could download them to the QA2D folder using gsutil:
```
gsutil cp gs://few-shot-fact-verification/QA2D_model/* ./QA2D/
```
### **Requirements for QA-to-Claim Model**
```
* Python==3.7.3
* torch==1.7.1
* tqdm==4.49.0
* transformers==4.3.3
* stanza==1.1.1
* nltk==3.5
* scikit-learn==0.23.2
* sense2vec
```
### Inputs and outputs (main.py)

Our input data is constructed based on the original FEVER dataset. We use the claim and evidence pairs labeled with SUPPORTED and REFUTED. The data after preprocced is in claim_evidence_pairs.csv.

* main.py goes through the steps of QA generation, Entailment checking, QA-to-Claim Model. That is, the claim and evidence pairs would generate the declarative sentences.
* Example output of main.py is explanation_all.csv

### Generation of counterfactual examples

After going through main.py, we use counterfactual_generation.py to convert the declarative sentences to three forms (Affirmative, Negative, Mixed) of counterfactual explanations.

