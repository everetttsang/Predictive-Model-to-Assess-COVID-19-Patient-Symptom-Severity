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
    zip.printdir()

    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')

cd pdf_json/

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



for item in worklist:
    count=count+1
    #print(count )
    #print(item)
    with open(item) as l:
        temp = json.load(l)
        dictionary_list.append(temp)
    if len(dictionary_list)%1000==0:
        print(len(dictionary_list))
print("Done")
#df_final = pd.DataFrame.from_dict(dictionary_list)
#df_final
    #data= data.append(temp, ignore_index=True)

#data
