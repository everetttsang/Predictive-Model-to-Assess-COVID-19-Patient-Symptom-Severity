#!/usr/bin/env python
# coding: utf-8


import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display
from time import gmtime, strftime
from sagemaker.predictor import csv_serializer


# Define IAM role
role = get_execution_role()
prefix = 'sagemaker/DEMO-xgboost-dm'
containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest',
              'us-west-1': '632365934929.dkr.ecr.us-west-1.amazonaws.com/xgboost:latest'} # each region has its XGBoost container
my_region = boto3.session.Session().region_name # set the region of the instance
print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + containers[my_region] + " container for your SageMaker endpoint.")

from pyspark.sql import SQLContext
import pyspark
from pyspark import SparkContext
sc = SparkContext.getOrCreate()
import pyspark
from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql import SQLContext
import pandas as pd
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import re #import regex

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

from pyspark import SparkFiles
# importing required modules
from zipfile import ZipFile

# specifying the zip file name
file_name = "pdf_json.zip"

# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
    # printing all the contents of the zip file
    #zip.printdir()

    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')

cd pdf_json

#!/usr/bin/env python
# coding: utf-8


import json

def remove_error_info(d):
    if not isinstance(d, (dict, list)):
        return d
    if isinstance(d, list):
        return [remove_error_info(v) for v in d]
    return {k: remove_error_info(v) for k, v in d.items()
            if k not in {'start', 'end', 'ref_end', 'ref_spans', 'cite_spans', 'bib_entries'}}




filenames = open('files.txt', 'r')
Lines = filenames.readlines()
#print(Lines)
worklist=[]
for line in Lines:
    worklist.append(line.strip())

print(worklist)

for item in worklist:
    print("Opening: " + item)
    with open(item) as f:
        data = json.load(f)

    data = remove_error_info(data)
    json_string = json.dumps(data, indent = 4, sort_keys=True)
    #print(json_string)

    with open(item, 'w') as json_file:
      json.dump(data, json_file, indent =4, sort_keys=True)
print("Done")

import json
import pandas as pd
from pandas.io.json import json_normalize #package for flattening json in pandas df


#with open('0a00a6df208e068e7aa369fb94641434ea0e6070.json') as f:
#        data = json.load(f)
#data = pd.json_normalize(data)

filenames= open('files.txt', 'r')
Lines = filenames.readlines()
worklist=[]

for line in Lines:
    worklist.append(line.strip())
    #print(worklist)
#print(worklist)
print(len(worklist))
#print(worklist[134353])
#print(count)
count=0

#split list into 4 separate workloads
def splitlist(x):
    #[[0-33587, 33588-67175, 67176-100764, 100765-134352]]
    first = x[:33587]

    second =x[33588:67175]
    third = x[67176: 100764]
    fourth = x[100765:]
    liste = []
    liste.append(first)

    liste.append(second)

    liste.append(third)

    liste.append(fourth)

    #print(len(liste))
    return liste

#worklist = split_lists(worklist, 4)
#print(worklist)
worklist = splitlist(worklist)

dictionary_list = []
#---------------------------------------------------
#with open(worklist[0]) as l:
#    temp = json.load(l)
#print(temp)
#dictionary_list.append(temp)
#with open(worklist[1]) as m:
#    temp = json.load(m)
#print(temp)
#dictionary_list.append(temp)
#print(dictionary_list)
#df_final = pd.DataFrame.from_dict(dictionary_list)
#df_final
#print(df_final.at[0, 'body_text'])
#print(df_final.at[1, 'body_text'])
#---------------------------------------------------


for x in range(4):
    for item in worklist[x]:
        count=count+1
        #print(count )
        #print(item)
        with open(item) as l:
            temp = json.load(l)
            dictionary_list.append(temp)
        if len(dictionary_list)%1000==0:
            print(len(dictionary_list))
    if x==0:
        df_final = pd.DataFrame.from_dict(dictionary_list)
    else:
        df_final = df_final.append(dictionary_list, ignore_index=True)
    dictionary_list=[]
    print("Done")
#df_final = pd.DataFrame.from_dict(dictionary_list)
df_final
    #data= data.append(temp, ignore_index=True)

#data

#add empty column "count"
df_final["count"]=0
#df_final["freq_pn"]= float(0)
#df_final["freq_di"]= float(0)

df_final["pneumonia_count"]= float(0)
df_final["diabetes_count"]= float(0)
df_final["pregnant_count"]= float(0)
df_final["asthma_count"]= float(0)
df_final["copd_count"]= float(0)
df_final["immunosuppressant_count"]= float(0)
df_final["hypertension_count"]= float(0)
df_final["obesity_count"]= float(0)
df_final["ckd_count"]= float(0)
df_final["tobacco_count"]= float(0)
df_final["cardiovascular_count"]= float(0)
pattern = ["pneumonia","diabetes", "pregnancy", "chronic obstructive pulomonary disease", "copd","asthma", "immunosuppressant", "hypertension","obesity","chronic kidney disease", "tobacco", "cardiovascular"]
vals = 0
#
#split the value with a space, do this with every word that has a space, can do space detection for
#keywords that we do not know yet
pattern3_split = pattern[3].split(" ")
pattern9_split = pattern[9].split(" ")

#search through each article
for x in df_final.index: #341712
    #keywords count to search
    pneumonia_count = 0;
    diabetes_count = 0;
    pregnant_count = 0
    asthma_count = 0
    copd_count = 0
    immunosuppressant_count = 0
    ckd_count = 0
    hypertension_count = 0
    obesity_count = 0
    tobacco_count = 0
    cardiovascular_count = 0

    wordcount = 0

    split3_qual = 0
    split9_qual = 0

    #search through titles in metadata

    ##################################################################
    #preprossessing to clean up the text
    abstractCurr = df_final.at[x, 'metadata']['title']
    abstractSplit = abstractCurr.split(" ")
    for word in abstractSplit:
        wordcount += 1
        word = word.lower() #convert to lowercase
        word = str(word).replace('.', '') #delete periods because they would be an issue for matches
        if(pattern[0] in word): #pneumonia
            pneumonia_count += 1
        if(pattern[1] in word): #diabetes
            diabetes_count += 1
        if(pattern[2] in word): #pregnancy
            pregnant_count += 1
        if(word in pattern3_split): #check if the word is in a list
            #check if the qual is made for the previous words, and if it matches the last item
            if(split3_qual == (len(pattern3_split) - 1) and word == pattern3_split[-1]):
                copd_count += 1
                split3_qual = 0
            if(pattern3_split[split3_qual] == word): #determine if the first strings match
                split3_qual += 1
            else:
                split3_qual = 0 #reset if no match
        if(pattern[4] in word): #other way to say copd
            copd_count += 1
        if(pattern[5] in word): #asthma
            asthma_count += 1
        if(pattern[6] in word): #immunosuppressant
            immunosuppressant_count += 1
        if(pattern[7] in word): #hypertension
            hypertension_count += 1
        if(pattern[8] in word): #obesity
            obesity_count += 1
        if(word in pattern9_split):
            #print(word)
            if(split9_qual == (len(pattern9_split) - 1) and word == pattern9_split[-1]):
                ckd_count += 1
                split9_qual = 0
            if(pattern3_split[split9_qual] == word):
                split9_qual += 1
            else:
                split9_qual = 0
        if(pattern[10] in word): #hypertension
            tobacco_count += 1
        if(pattern[11] in word): #obesity
            cardiovascular_count += 1
    ##################################################################
    #search through abstract
    for y in range(len(df_final.at[x, 'abstract'])):
        #print(df_final.at[1, 'abstract'][y]['text'])

        #preprossessing to clean up the text
        abstractCurr = df_final.at[x, 'abstract'][y]['text']
        abstractSplit = abstractCurr.split(" ")
        #print(abstractSplit)
        for word in abstractSplit:
            wordcount += 1
            if(pattern[0] in word):
                pneumonia_count += 1
            if(pattern[1] in word):
                diabetes_count += 1


    #search through body text
    for y in range(len(df_final.at[x, 'body_text'])):
        #print(df_final.at[1, 'abstract'][y]['text'])

        #preprossessing to clean up the text
        abstractCurr = df_final.at[x, 'body_text'][y]['text']
        abstractSplit = abstractCurr.split(" ")
        #print(abstractSplit)
        for word in abstractSplit:
            wordcount += 1
            if(pattern[0] in word):
                pneumonia_count += 1
            if(pattern[1] in word):
                diabetes_count += 1

    #search through "back_matter"
    for y in range(len(df_final.at[x, 'back_matter'])):
        #print(df_final.at[1, 'abstract'][y]['text'])

        #preprossessing to clean up the text
        abstractCurr = df_final.at[x, 'back_matter'][y]['text']
        abstractSplit = abstractCurr.split(" ")
        #print(abstractSplit)
        for word in abstractSplit:
            wordcount += 1
            if(pattern[0] in word):
                pneumonia_count += 1
            if(pattern[1] in word):
                diabetes_count += 1

    #search through captions to figures
    for y in df_final.at[x, 'ref_entries']:
        #preprossessing to clean up the text
        abstractCurr = df_final.at[x, 'ref_entries'][y]['text']
        abstractSplit = abstractCurr.split(" ")
        #print(abstractSplit)
        for word in abstractSplit:
            wordcount += 1
            if(pattern[0] in word):
                pneumonia_count += 1
            if(pattern[1] in word):
                diabetes_count += 1

    #pneumonia_freq = float(pneumonia_count) / float(wordcount-pneumonia_count);
    #diabetes_freq = float(diabetes_count) / float(wordcount);

#     df.at[x, 'count'] = count
    #df_final.at[x, 'freq_pn'] = pneumonia_freq
    #df_final.at[x, 'freq_di'] = diabetes_freq
    df_final.at[x, 'count'] = wordcount
    df_final.at[x, 'pneumonia_count']= pneumonia_count
    df_final.at[x, 'diabetes_count']= diabetes_count
    df_final.at[x, 'pregnant_count']= pregnant_count
    df_final.at[x, 'asthma_count']= asthma_count
    df_final.at[x, 'copd_count']= copd_count
    df_final.at[x, 'immunosuppressant_count']= immunosuppressant_count
    df_final.at[x, 'hypertension_count']= hypertension_count
    df_final.at[x, 'obesity_count']= obesity_count
    df_final.at[x, 'ckd_count']= ckd_count
    df_final.at[x, 'tobacco_count']= tobacco_count
    df_final.at[x, 'cardiovascular_count']= cardiovascular_count

df_final

#Runtime + dataparse: 10:54
keywordList = ['pneumonia_count', 'diabetes_count','pregnant_count','pregnant_count','asthma_count','copd_count','immunosuppressant_count','hypertension_count','obesity_count','ckd_count','tobacco_count','cardiovascular_count']

for keyword in keywordList:
    maxURL = []
    prevFreqs = []
    prevIndexes = []
    articleName = []
    currFreq = 0
    maxFreq = 0
    maxIndex = 0
    for i in range(0,10):
        for x in df_final.index: #341712
            if(df_final.at[x,keyword] > maxFreq and prevIndexes.count(x) == 0):
                maxFreq = df_final.at[x,keyword]
                maxIndex = x
        prevFreqs.append(maxFreq)
        prevIndexes.append(maxIndex)
        maxFreq = 0

    #print(prevFreqs)
    #print(prevIndexes)
    for j in prevIndexes:
        articleName.append(df_final.at[j, 'metadata']['title'])

    print('For {}: --------'.format(keyword))
    for ten in range(len(articleName)):
        print('[{}]: {}'.format(ten+1, articleName[ten]))
    print('------------------')
