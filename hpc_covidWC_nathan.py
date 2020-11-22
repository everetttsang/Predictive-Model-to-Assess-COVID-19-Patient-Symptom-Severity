#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[9]:


abstract_metadata="https://hpcdata.s3-us-west-1.amazonaws.com/metadata.csv"
sc.addFile(abstract_metadata)
sqlContext = SQLContext(sc)

df = pd.read_csv(SparkFiles.get("metadata.csv"), header = 0)
df = df.drop(['sha', 'cord_uid', 'doi', 'source_x', 'pmcid', 'pubmed_id', 'source_x', 'license', 'publish_time', 'journal','who_covidence_id', 'mag_id', 'arxiv_id', 'pdf_json_files', 'pmc_json_files', 's2_id'], axis=1)
df = df.astype({'abstract': 'string'})


#drop all rows containing abstract="<NA>"
df = df.dropna(subset=['abstract'])

df.index.name="index"

#add empty column "count"
df["count"]=0
df["freq_pn"]= float(0)
df["freq_di"]= float(0)

print(df.dtypes)
#print(df.at[0, 'abstract'])


# In[13]:


pattern = ["pneumonia","diabetes"]
vals = 0

for x in df.index: #341712
    #keywords count to search
    pneumonia_count = 0;
    diabetes_count = 0;
    wordcount = 0
    
    #preprossessing to clean up the text
    abstractCurr = df.at[x, 'abstract']
    abstractSplit = abstractCurr.split(" ")
    for word in abstractSplit:
        wordcount += 1
        if(pattern[0] in word):
            pneumonia_count += 1
        if(pattern[1] in word):
            diabetes_count += 1
        
            
    pneumonia_freq = float(pneumonia_count) / float(wordcount-pneumonia_count);
    diabetes_freq = float(diabetes_count) / float(wordcount);
    
#     df.at[x, 'count'] = count
    df.at[x, 'freq_pn'] = pneumonia_freq
    df.at[x, 'freq_di'] = diabetes_freq
    
df


# In[20]:


maxURL = []
prevFreqs = []
prevIndexes = []
currFreq = 0
maxFreq = 0
maxIndex = 0
for i in range(0,10):
    for x in df.index: #341712
        if(df.at[x,'freq_pn'] > maxFreq and prevIndexes.count(x) == 0):
            maxFreq = df.at[x,'freq_pn']
            maxIndex = x
    prevFreqs.append(maxFreq)
    prevIndexes.append(maxIndex)
    maxFreq = 0
        
print(prevFreqs)
print(prevIndexes)


# In[ ]:




