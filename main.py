#!/usr/bin/env python
# coding: utf-8
import math
import re
import os
import spacy
import json
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from Questgen import main
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# read fever dataset from csv, including claim and evidence columns
claim_evidence_df = pd.read_csv('./claim_evidence_pairs.csv', encoding='utf8')

# load MCQ/ FAQ questions generator model
qg = main.QGen()

questions = set()
qpayload = {
            "input_text": "Katie Stevens is Australian."
        }

out_dir = "questions/"
os.makedirs("questions/", exist_ok=True)
claim_question = {}
count=0
for claim in tqdm(claim_evidence_df['claim'][count:500]):
    payload = copy.deepcopy(qpayload)
    payload['input_text'] = claim

    # generate MCQ questions
    output = qg.predict_mcq(payload)
    if len(output) != 0:
        for q in output['questions']:
            questions.add(q['question_statement'])
    # generate FAQ questions
    output = qg.predict_shortq(payload)
    if len(output) != 0:
        for q in output['questions']:
            questions.add(q['Question'])
    try:
        claim_question[str(count)+"_"+claim] = list(dict.fromkeys(questions))
    except:
        print(claim, list(dict.fromkeys(questions)))
    questions = set()
    count=count+1
    div,mod = divmod(count, 100)
    if mod == 0:
        with open(out_dir+"question_"+str(div)+".json", "w") as outfile: 
            json.dump(claim_question, outfile)
            claim_question = {}
with open(out_dir+"question_"+str(div+1)+".json", "w") as outfile: 
    json.dump(claim_question, outfile)
    claim_question = {}


Questions_df = pd.read_json("questions/question_1.json", orient ='index', encoding='utf8')
for i in range(2,6):
    df1 = pd.read_json("questions/question_"+str(i)+".json", orient ='index', encoding='utf8')
    Questions_df = pd.concat([Questions_df, df1])
Questions_df = Questions_df.reset_index()
question_num = len(Questions_df.columns)-1
question_col = ['q1','q2','q3','q4','q5','q6','q7','q8','q9','q10','q11','q12','q13','q14','q15','q16','q17']
Questions_df.columns = ['claim'] + question_col[:question_num]
Questions_df["claim"] = Questions_df["claim"].str.replace("“","").replace("”","").replace(r"\d+_","", regex=True)


qc_df = Questions_df
qc_df = qc_df.fillna("None")

answer = main.AnswerPredictor()

out_dir = "QAs/"
os.makedirs(out_dir, exist_ok=True)

QA={
    "question": "",
    "answer":""
}

payload = {
    "input_text" : '''At its greatest extent , the state expanded into territories that today comprise most of Iran , Iraq , Armenia , Azerbaijan , Georgia , Turkmenistan , Turkey , western Afghanistan , and southwestern Pakistan .''',
    "input_question" : "Who is the husband of Randy Couture?"

}

output_dic = {}
Answer_dic = {}
count = 0
for i, evidence in tqdm(enumerate(claim_evidence_df["evidence"][count:500])):
    # truncate the text into the length of 512
    payload["input_text"] = evidence[:512]
    Answer_row = []
    for q in qc_df.iloc[count+i][1:]:#q1 start from 1th column
        payload["input_question"] = q
        if q == "None":
            a = "None"
        else:
            a = answer.predict_answer(payload)
        copy_QA = copy.deepcopy(QA)
        copy_QA["question"] = q
        copy_QA["answer"] = a
        
        Answer_row.append(copy_QA)
        copy_Answer_dic = copy.deepcopy(Answer_dic)
        copy_Answer_dic["evidence"] = evidence
        copy_Answer_dic["claim"] = qc_df.iloc[count+i]["claim"]
        copy_Answer_dic["QA"] = Answer_row
        
    output_dic[count+i] = copy_Answer_dic

    div, mod = divmod(count+i+1,100)
    if mod == 0:
        with open(out_dir+"output_dic_"+str(div)+".json", "w") as outfile: 
            json.dump(output_dic, outfile)
            output_dic = {}
with open(out_dir+"output_dic_"+str(div+1)+".json", "w") as outfile: 
    json.dump(output_dic, outfile)
    output_dic = {}

df = pd.read_json("QAs/output_dic_1.json", orient ='index')
for i in range(2,6):
    df1 = pd.read_json("QAs/output_dic_"+str(i)+".json", orient ='index')
    df = pd.concat([df, df1])

#Entailment Checker (picking the best answer)
tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

answers = []
no_accpect = ["TRUE", "yes", "FALSE", "True", "Yes", "False", "No", "None"]
os.makedirs("best_answers/", exist_ok=True)
for i, claim in tqdm(enumerate(df["claim"][:])):
    ans_strings = []
    in_strings = []
    #in_strings_backup = []
    for qa in df.iloc[i]["QA"]:
        if qa["question"] != None and qa["answer"] not in no_accpect:
            string = claim + "</s></s>" + qa["answer"]
            ans_strings.append(qa["answer"])
            in_strings.append(string)
    if len(in_strings) !=0:
        inputs = tokenizer(in_strings, padding=True,return_tensors="pt")
        outputs = model(**inputs)
        print(in_strings)
        print(outputs)
        #pred = outputs.logits.max(1).indices
        outputs_np = outputs[0].detach().numpy()
        max_contra_id = np.argmax(outputs_np.T[0], axis=0)
        #    answers.append(df.iloc[i]["QA"][max_contra_id]["answer"])
        answers.append(ans_strings[max_contra_id])
    else:
        answers.append(" ")
    div, mod = divmod(i+1,100)
#    answers.append(df.iloc[i]["QA"][max_contra_id]["answer"])
    if mod == 0:
        answers_df = pd.DataFrame(answers)
        answers = []
        answers_df.to_csv("best_answers/output_"+str(div)+".csv")
answers_df = pd.DataFrame(answers)
answers = []
answers_df.to_csv("best_answers/output_"+str(div+1)+".csv")

df = pd.read_json("QAs/output_dic_1.json", orient ='index')
for i in range(2,6):
    df1 = pd.read_json("QAs/output_dic_"+str(i)+".json", orient ='index')
    df = pd.concat([df, df1])

df2 = pd.read_csv("best_answers/output_1.csv", names=['best answer'], header=0)
for i in range(2,6):
    df1 = pd.read_csv("best_answers/output_"+str(i)+".csv", names=['best answer'], header=0)
    df2 = pd.concat([df2, df1], ignore_index=True)

df = df.join(df2)
QA_df = df

for i in question_num:
    questions = []
    answers = []
    for qa in QA_df["QA"]:
        answers.append(qa[i]["answer"])
        questions.append(qa[i]["question"])
    QA_df["Q"+str(i+1)] = questions
    QA_df["A"+str(i+1)] = answers


model_args = Seq2SeqArgs()
model_args.max_length = 1000

QA2D_model = Seq2SeqModel(
            encoder_decoder_type="bart", 
            encoder_decoder_name="./QA2D_model",
            cuda_device=0,
            args=model_args)


r = re.compile(r'[.]$')
df = df.replace(np.nan, '', regex=True)

d = {}
for i in question_num:
    d["Q"+str(i+1)] = "A"+str(i+1)

counta =0
output = []
results = []
wrong_id = []
nlp = spacy.load("en_core_web_sm")
for index, row in df.iterrows():
    counta =0
    for answer_col in d:
        if row["best answer"] == row[answer_col]:
            #if row[d[answer_col]] == "":
            doc = nlp(row["best answer"])
            sent = False
            for token in doc:
                if token.dep_ == "nsubj":
                    sent = True
                    break

            if r.search(row["best answer"]) != None and sent == True:
                results.append(row["best answer"].replace("Yes, ", "").capitalize())
            else:
                results.append(QA2D_model.predict([row[d[answer_col]] + ' [SEP] ' + row["best answer"]])[0])
            break

results_df = pd.DataFrame(results)
QA_df.join(results_df).to_csv("explanation_all.csv")

