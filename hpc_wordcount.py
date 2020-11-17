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


abstract_metadata="https://hpcdata.s3-us-west-1.amazonaws.com/metadata.csv"
sc.addFile(abstract_metadata)
sqlContext = SQLContext(sc)

df = pd.read_csv(SparkFiles.get("metadata.csv"), header = 0)
df = df.drop(['sha', 'cord_uid', 'doi', 'source_x', 'pmcid', 'pubmed_id', 'source_x', 'license', 'publish_time', 'journal','who_covidence_id', 'mag_id', 'arxiv_id', 'pdf_json_files', 'pmc_json_files', 's2_id'], axis=1)
df = df.astype({'abstract': 'string'})

#add empty column "count"
df["count"]=0

print(df.dtypes)
#print(df.at[0, 'abstract'])

pattern = 'Mycoplasma'
for x in range(341713):
    count =0;

    for match in re.finditer(pattern, df.at[x, 'abstract']):
        print(match)
        count+= 1
        df.at[x, 'count'] = count

#print(df.at[0,'abstract'])




df
