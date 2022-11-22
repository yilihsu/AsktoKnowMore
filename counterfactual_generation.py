#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import nltk
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import re


df = pd.read_csv('./explanation_all.csv') 

change_dic = {" has ": " HAS NOT ",
 " was ": " WAS NOT ",
" am ": " AM NOT ",
" is ": " IS NOT ",
" are ": " ARE NOT ",
" will ": " WILL NOT ",
" can ": " CAN NOT ",
" have ": " HAVE NOT ",
" were ": " WERE NOT ",
" did ": " DID NOT ",
" does ": " DOES NOT ",
"n't": ""}



# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")
output = []

for i, r in enumerate(df["claim"]):
    change = 0
    if " not " in r:
        output.append(r.replace(" not ", " "))
        change = 1
        continue
    elif " only " in r:
        output.append(r.replace(" only ", " NOT only "))
        change = 1
        continue
    elif " yet to be " in r:
        output.append(r.replace(" yet to be ", " "))
        change = 1
        continue
    elif " yet to " in r:
        output.append(r.replace(" yet to ", " "))
        change = 1
        continue
    elif " never " in r:
        output.append(r.replace(" never ", " "))
        change = 1
        continue
    elif " always " in r:
        output.append(r.replace(" always ", " NOT always "))
        change = 1
        continue
    elif " exclusively " in r:
        output.append(r.replace(" exclusively ", " NOT exclusively "))
        change = 1
        continue
    else:
        for key in change_dic.keys():
            if change == 0 and key in r:
                output.append(r.replace(key, change_dic[key]))
                change = 1
    doc = nlp(r)
    if change == 0:
        for token in doc:
            if token.tag_ == "VBZ":
                output.append(r.replace(token.text, "DOES NOT " + token.lemma_))
                change = 1
                break
            elif token.tag_ == "VBD" or token.tag_ == "VBN":
                output.append(r.replace(token.text,  "DID NOT " + token.lemma_))
                change = 1
                break
            elif token.tag_ == "VBP":
                output.append(r.replace(token.text, "DO NOT " + token.lemma_))
                change = 1
                break
    if change == 0:
        for token in doc:
            if token.tag_ == "RB":
                output.append(r.replace(token.text, "is NOT"))
                change =1
                break
    if change ==0:
        output.append(r)
        print(r)
df["output"] = output

#generate counterfactual form

nlp = spacy.load("en_core_web_sm")

upper_dic = {
    "english": "English",
    "indian": "Indian",
    "american": "American",
    "french": "French",
    "canadian": "Canadian"
}

def get_ngrams(text, n):
    n_grams = ngrams(word_tokenize(text), n)
    return [' '.join(grams) for grams in n_grams]
def replace(old, new, str, caseinsentive = False):
    if caseinsentive:
        return str.replace(old, new)
    else:
        return re.sub(re.escape(old), new, str, flags=re.IGNORECASE)
    
counter_factuals_1=[]
counter_factuals_2=[]
counter_factuals_3=[]

for id, row in df.iterrows():
    explanation = row['0']
    neg_claim = row['output']
    claim = row['claim'].replace("”","").replace("“","").replace('"',"").replace('.',"").replace('(',"").replace(')',"")
    
    # nationality to uppercase
    for key in upper_dic.keys():
        if key in explanation:
            explanation = explanation.replace(key, upper_dic[key])

    claim = claim.replace("Says","").strip()
    grams = [4,3,2]
    for gram in grams:
        current_grams = get_ngrams(re.sub(r'[^\w\s]','',explanation),gram)
        for fg in current_grams:
            insensitive_re = re.compile(u"^"+fg, re.IGNORECASE)
            claim = insensitive_re.sub("",claim).strip()

    if claim == "":
        counter_factual = "If we were to say '" + explanation + "' the claim would be correct."
    else:    
        counter_factual = "If we were to say '" + explanation.replace(".","") +"' instead of '" + claim + "', the claim would be correct."

    tokens = word_tokenize(neg_claim.lower())
    tags = nltk.pos_tag(tokens)
    
    removed = False

    for tag in tags:
        if tag[1] == "VBZ" or tag[1] == "VBD" or tag[1] == "VBP" or tag[1] == "VBN":
            if re.match(r"(.*) "+tag[0], neg_claim.lower()) != None and re.match(r"(.*) "+tag[0], explanation.lower()) != None:
                sub_ng_claim = re.match(r"(.*) "+tag[0], neg_claim.lower()).group(1)
                sub_exp = re.match(r"(.*) "+tag[0], explanation.lower()).group(1)
                if sub_ng_claim.lower() == sub_exp.lower():
                    sub_explanation = replace(sub_exp+" "+tag[0],"",explanation).strip()
                    removed = True
                    for key in change_dic.keys():
                        if key in sub_explanation:
                            sub_explanation = sub_explanation.replace(key,"").strip()
                            removed = True
                            break
                    break
                    
    if removed == False:
        exp = nlp(explanation)
        for token in exp:
            if token.dep_ == "nsubj":
                token1 = token.text
            if token.dep_ == "ROOT":
                root = token.text
        ng_clm = nlp(neg_claim)
        for token in ng_clm:
            if token.dep_ == "nsubj":
                token2 = token.text
        
        #remove same subjects
        if token1.lower()==token2.lower():
            if re.match(r"(.*) "+token1, explanation[:25]) != None:
                sub_exp = re.match(r"(.*) "+token1, explanation[:25]).group(1)
                sub_explanation = replace(sub_exp+" "+token1,"",explanation[:25])+explanation[25:].strip()
                removed = True
                for key in change_dic.keys():
                    if key in sub_explanation:
                        sub_explanation = sub_explanation.replace(key,"").strip()
                        break
                #print("===1")
            elif re.match(token1, explanation[:25]) != None:
                sub_explanation = replace(token1,"",explanation[:25])+explanation[25:].strip()
                removed = True
                for key in change_dic.keys():
                    if key in sub_explanation:
                        sub_explanation = sub_explanation.replace(key,"").strip()
                        break
                #print("===2")

    if removed == False:
        sub_explanation = explanation
    
    scores = rouge.get_scores(sub_explanation,neg_claim)
    high_rouge = 0
    if round(rouge.get_scores(sub_explanation,neg_claim)[0]['rouge-1']['f'], 2) >= 0.75:
        high_rouge = 1
    elif round(rouge.get_scores(explanation,neg_claim)[0]['rouge-1']['f'], 2) >= 0.75:
        high_rouge = 1
        
    if high_rouge == 1:
        #print(sub_explanation,neg_claim)
        counter_factual2 = "If we were to say '"+ neg_claim + "', the claim would be correct."
    elif "not" in neg_claim or "NOT" in neg_claim:
        counter_factual2 = "If we were to say '" + neg_claim.replace(".","").replace(",","") +" but " + sub_explanation + "', the claim would be correct."
    else:
        counter_factual2 = "If we were to say '" + neg_claim.replace(".","").replace(",","") +" and " + sub_explanation + "', the claim would be correct."

    counter_factual3 = "If we were to say '" + row['claim'] +"' but say '" + explanation + "', the claim would be correct."

    counter_factuals_1.append(counter_factual)
    counter_factuals_2.append(counter_factual2)
    counter_factuals_3.append(counter_factual3)
df['counter_factual_Affirmative'] = counter_factuals_1
df['counter_factual_Negative'] = counter_factuals_2
df['counter_factual_Mixed'] = counter_factuals_3
df.to_csv("./output.csv")
