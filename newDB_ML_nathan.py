#!/usr/bin/env python
# coding: utf-8

# In[12]:


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
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'} # each region has its XGBoost container
my_region = boto3.session.Session().region_name # set the region of the instance
print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + containers[my_region] + " container for your SageMaker endpoint.")


# In[2]:


#from pyspark.sql import SQLContext
import pyspark
from pyspark import SparkContext
sc = SparkContext.getOrCreate()


# In[13]:



#retrieve the CSV from S3
try:
  urllib.request.urlretrieve ("https://projectdatasetcovid.s3-us-west-2.amazonaws.com/201029COVID19MEXICO.csv","201029COVID19MEXICO.csv")
  print('Success: 201029COVID19MEXICO.csv')
except Exception as e:
  print('Data load error: ',e)



import pyspark
from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql import SQLContext

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

#add the file to the spark context

#from pyspark.sql import SQLContext
url = "201029COVID19MEXICO.csv"
from pyspark import SparkFiles
sc.addFile(url)
sqlContext = SQLContext(sc)


# In[14]:


#read the csv and create a dataframe
df = sqlContext.read.csv(SparkFiles.get("201029COVID19MEXICO.csv"), header=True, inferSchema= True)


df.printSchema()

#group and print the count of type of patient 
# 1 = outpatient, 2 = inpatient 
df.groupBy("TIPO_PACIENTE").count().sort("count",ascending=True).show()


# In[15]:


import pyspark.sql.functions as F
#df1 = df.withColumnRenamed("TIPO_PACIENTE","Patient_Type")\.withColumnRenamed("SEXO","Sex")

#convert the spanish column labels to english - now in df1
df1 = df.selectExpr("TIPO_PACIENTE as PATIENT_TYPE",
                   "SEXO as SEX",
                   "Age as AGE",
                   "INTUBADO as INTUBED",
                    "NEUMONIA as NEUMONIA",
                     "Pregnant as PREGNANT",
                   "DIABETES as DIABETES",
                  "EPOC as EPOC",
                   "ASMA as ASTHMA",
                   "INMUSUPR as INMUSUPR",
                   "HIPERTENSION as HYPERTENSION",
                   "CARDIOVASCULAR as CARDIOVASCULAR",
                   "OBESIDAD as OBESITY",
                   "RENAL_CRONICA as RENAL_CRONIC",
                   "OTRA_COM as OTHER_DISEASE",
                  "TABAQUISMO as TOBACCO",
                   "OTRO_CASO as CONTACT_OTHER_COVID",
                   "CLASIFICACION_FINAL as CLASSIFICACION_FINAL",
                   "UCI as ICU",
                   "RESULTADO_LAB as LAB_RESULT",
                    "FECHA_DEF as DEATH"
                  )


df1.printSchema()


# In[16]:


df1.show(5)

from pyspark.sql.functions import col, when

#convert the data from their format to an easier to read format 

#If yes ->1, No -> 0 and Missing/NA -> np.nan
# df2 = df1.withColumn("PREGNANT",when(col("PREGNANT") == "2", 0).when(col("PREGNANT") == "1", 1).otherwise(np.nan))
# df2 = df2.withColumn("NEUMONIA",when(col("NEUMONIA") == "2", 0).when(col("NEUMONIA") == "1", 1).otherwise(np.nan))
# df2 = df2.withColumn("INTUBED",when(col("INTUBED") == "2", 0).when(col("INTUBED") == "1", 1).otherwise(np.nan))
# df2 = df2.withColumn("DIABETES",when(col("DIABETES") == "2", 0).when(col("DIABETES") == "1", 1).otherwise(np.nan))
# df2 = df2.withColumn("EPOC",when(col("EPOC") == "2", 0).when(col("EPOC") == "1", 1).otherwise(np.nan))
# df2 = df2.withColumn("ASTHMA",when(col("ASTHMA") == "2", 0).when(col("ASTHMA") == "1", 1).otherwise(np.nan))
# df2 = df2.withColumn("INMUSUPR",when(col("INMUSUPR") == "2", 0).when(col("INMUSUPR") == "1", 1).otherwise(np.nan))
# df2 = df2.withColumn("HYPERTENSION",when(col("HYPERTENSION") == "2", 0).when(col("HYPERTENSION") == "1", 1).otherwise(np.nan))
# df2 = df2.withColumn("OTHER_DISEASE",when(col("OTHER_DISEASE") == "2", 0).when(col("OTHER_DISEASE") == "1", 1).otherwise(np.nan))
# df2 = df2.withColumn("CARDIOVASCULAR",when(col("CARDIOVASCULAR") == "2", 0).when(col("CARDIOVASCULAR") == "1", 1).otherwise(np.nan))
# df2 = df2.withColumn("OBESITY",when(col("OBESITY") == "2", 0).when(col("OBESITY") == "1", 1).otherwise(np.nan))
# df2 = df2.withColumn("RENAL_CRONIC",when(col("RENAL_CRONIC") == "2", 0).when(col("RENAL_CRONIC") == "1", 1).otherwise(np.nan))
# df2 = df2.withColumn("TOBACCO",when(col("TOBACCO") == "2", 0).when(col("TOBACCO") == "1", 1).otherwise(np.nan))
# #leave out contact_other_covid
# df2 = df2.withColumn("CONTACT_OTHER_COVID",when(col("CONTACT_OTHER_COVID") == "2", 0).when(col("CONTACT_OTHER_COVID") == "1", 1).otherwise(np.nan))
# df2 = df2.withColumn("ICU",when(col("ICU") == "2", 0).when(col("ICU") == "1", 1).otherwise(np.nan))

# #we might want to change the 9999-99-99 to 0 instead of nan and dead to 1
# df2 = df2.withColumn("DEATH",when(col("DEATH") == "9999-99-99", np.nan).otherwise("DEAD"))

#assume missing data = no -> 0
df2 = df1.withColumn("PREGNANT",when(col("PREGNANT") == "2", 0).when(col("PREGNANT") == "1", 1).otherwise(0))
df2 = df2.withColumn("NEUMONIA",when(col("NEUMONIA") == "2", 0).when(col("NEUMONIA") == "1", 1).otherwise(0))
df2 = df2.withColumn("INTUBED",when(col("INTUBED") == "2", 0).when(col("INTUBED") == "1", 1).otherwise(0))
df2 = df2.withColumn("DIABETES",when(col("DIABETES") == "2", 0).when(col("DIABETES") == "1", 1).otherwise(0))
df2 = df2.withColumn("EPOC",when(col("EPOC") == "2", 0).when(col("EPOC") == "1", 1).otherwise(0))
df2 = df2.withColumn("ASTHMA",when(col("ASTHMA") == "2", 0).when(col("ASTHMA") == "1", 1).otherwise(0))
df2 = df2.withColumn("INMUSUPR",when(col("INMUSUPR") == "2", 0).when(col("INMUSUPR") == "1", 1).otherwise(0))
df2 = df2.withColumn("HYPERTENSION",when(col("HYPERTENSION") == "2", 0).when(col("HYPERTENSION") == "1", 1).otherwise(0))
df2 = df2.withColumn("OTHER_DISEASE",when(col("OTHER_DISEASE") == "2", 0).when(col("OTHER_DISEASE") == "1", 1).otherwise(0))
df2 = df2.withColumn("CARDIOVASCULAR",when(col("CARDIOVASCULAR") == "2", 0).when(col("CARDIOVASCULAR") == "1", 1).otherwise(0))
df2 = df2.withColumn("OBESITY",when(col("OBESITY") == "2", 0).when(col("OBESITY") == "1", 1).otherwise(0))
df2 = df2.withColumn("RENAL_CRONIC",when(col("RENAL_CRONIC") == "2", 0).when(col("RENAL_CRONIC") == "1", 1).otherwise(0))
df2 = df2.withColumn("TOBACCO",when(col("TOBACCO") == "2", 0).when(col("TOBACCO") == "1", 1).otherwise(0))
#leave out contact_other_covid
df2 = df2.withColumn("CONTACT_OTHER_COVID",when(col("CONTACT_OTHER_COVID") == "2", 0).when(col("CONTACT_OTHER_COVID") == "1", 1).otherwise(0))
df2 = df2.withColumn("ICU",when(col("ICU") == "2", 0).when(col("ICU") == "1", 1).otherwise(0))

#we might want to change the 9999-99-99 to 0 instead of nan and dead to 1
df2 = df2.withColumn("DEATH",when(col("DEATH") == "9999-99-99", 0).otherwise("DEAD"))

df2.show(10);


# In[17]:


#Patient_type 
#1-> Hospitalised
#2->Home Care

df2 = df2.filter("LAB_RESULT == 1")

df2.groupBy("PATIENT_TYPE").count().sort("count",ascending=True)

#If Death =1 , then pateint died
df2.groupBy("DEATH").count().sort("count",ascending=True)

#see all rows when pateint died
df2.filter("DEATH == 'DEAD'").show(10)

#Sex -> Male =2 , so change the PREGNANT for MALE as '0'
df2 = df2.withColumn("PREGNANT",when(col("SEX") == "2", 0))
df2.filter("SEX == 2").show(10)

#converts the nan to unknown
df_unknown_imputed = df2.fillna("Unknown")

df_unknown_imputed.cache()

#number of entries on this dataset
df_unknown_imputed.count()


# In[18]:


#function to return the features and label with relation to the category, continous, and label columns of DF
def get_dummy(df,categoricalCols,continuousCols,labelCol):

    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.sql.functions import col

    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c),handleInvalid="keep")
                 for c in categoricalCols ]
    
    #val categoryIndexerModel = new StringIndexer().setInputCol("category").setOutputCol("indexedCategory").setHandleInvalid("keep") 

    # default setting: dropLast=True
    encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),
                 outputCol="{0}_encoded".format(indexer.getOutputCol()))
                 for indexer in indexers ]

    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]
                                + continuousCols, outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler])

    model=pipeline.fit(df)
    data = model.transform(df)

    data = data.withColumn('label',col(labelCol))

    return data.select('features','label')


# In[19]:




from pyspark.sql.types import *

#convert the data of death from DEATH and NaN to 1 and 0
df_unknown_imputed = df_unknown_imputed.withColumn("DEATH",when(col("DEATH") == "DEAD", 1).otherwise(0))
# Change column type to double
df_unknown_imputed = df_unknown_imputed.withColumn("DEATH", df_unknown_imputed["DEATH"].cast(DoubleType()))

#take the caracteristics, and symptoms as the category columns 'features'
catcols = ['SEX','NEUMONIA', 'PREGNANT','DIABETES', 'EPOC', 'ASTHMA', 'INMUSUPR','HYPERTENSION','OTHER_DISEASE',
           'CARDIOVASCULAR','OBESITY','RENAL_CRONIC','TOBACCO']

#these are results from the symptoms 
num_cols = ['DEATH', 'ICU','INTUBED']
# num_cols = ['DEATH', 'ICU']
#try to predict this (outpatient vs inpatient), can predict death and ICU later
labelCol = 'PATIENT_TYPE'

data = get_dummy(df_unknown_imputed,catcols,num_cols,labelCol)
data.show(5)


# In[20]:


from pyspark.ml.feature import StringIndexer
# Index labels, adding metadata to the label column
labelIndexer = StringIndexer(inputCol='label',
                             outputCol='indexedLabel').fit(data)
labelIndexer.transform(data)


from pyspark.ml.feature import VectorIndexer
# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =VectorIndexer(inputCol="features",outputCol="indexedFeatures",maxCategories=4).fit(data)
featureIndexer.transform(data)



# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

#breakdown of the label data that we willbe predicting 
print("Data")
data.groupBy("label").count().sort("count",ascending=True).show()
print("Training data")
trainingData.groupBy("label").count().sort("count",ascending=True).show()
print("Test data")
testData.groupBy("label").count().sort("count",ascending=True).show()


# In[21]:


############################################################
#logistic regression 

from pyspark.ml.classification import LogisticRegression
logr = LogisticRegression(featuresCol='indexedFeatures', labelCol='indexedLabel')

from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString,StringIndexer, VectorIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)


# In[22]:


# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, logr,labelConverter])


# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)
#java.lang.IllegalArgumentException: requirement failed: init value should <= bound

# Make predictions.
predictions = model.transform(testData)
# Select example rows to display.
predictions.select("features","label","predictedLabel").show(5)




from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Logistic regression ML accuracy to predict inpatient vs outpatient")
print(accuracy)


# In[ ]:




