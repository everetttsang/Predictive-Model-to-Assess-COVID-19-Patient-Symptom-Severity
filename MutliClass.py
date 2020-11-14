#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import libraries
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


bucket_name = 'projectdatasetcovid' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
s3 = boto3.resource('s3')
try:
    if  my_region == 'us-east-1':
      s3.create_bucket(Bucket=bucket_name)
    else: 
      s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)


# In[3]:


try:
  urllib.request.urlretrieve ("https://projectdatasetcovid.s3-us-west-2.amazonaws.com/201029COVID19MEXICO.csv","201029COVID19MEXICO.csv")
  print('Success: 201029COVID19MEXICO.csv')
except Exception as e:
  print('Data load error: ',e)


# In[4]:


import pyspark
from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql import SQLContext

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)


# In[5]:


#from pyspark.sql import SQLContext
url = "201029COVID19MEXICO.csv"
from pyspark import SparkFiles
sc.addFile(url)
sqlContext = SQLContext(sc)


# In[6]:


df = sqlContext.read.csv(SparkFiles.get("201029COVID19MEXICO.csv"), header=True, inferSchema= True)


# In[7]:


df.printSchema()


# In[8]:


import pyspark.sql.functions as F
#df1 = df.withColumnRenamed("TIPO_PACIENTE","Patient_Type")\.withColumnRenamed("SEXO","Sex")

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


# In[9]:


from pyspark.sql.functions import col, when

#changed DEATH TO 1 or 0

#If yes ->1, No -> 0 and Missing/NA -> 2
df2 = df1.withColumn("PREGNANT",when(col("PREGNANT") == "2", 0).when(col("PREGNANT") == "1", 1).when(col("SEX") == "2", 0).otherwise(np.nan))
df2 = df2.withColumn("NEUMONIA",when(col("NEUMONIA") == "2", 0).when(col("NEUMONIA") == "1", 1).otherwise(np.nan))
df2 = df2.withColumn("INTUBED",when(col("INTUBED") == "2", 0).when(col("INTUBED") == "1", 1).otherwise(np.nan))
df2 = df2.withColumn("DIABETES",when(col("DIABETES") == "2", 0).when(col("DIABETES") == "1", 1).otherwise(np.nan))
df2 = df2.withColumn("EPOC",when(col("EPOC") == "2", 0).when(col("EPOC") == "1", 1).otherwise(np.nan))
df2 = df2.withColumn("ASTHMA",when(col("ASTHMA") == "2", 0).when(col("ASTHMA") == "1", 1).otherwise(np.nan))
df2 = df2.withColumn("INMUSUPR",when(col("INMUSUPR") == "2", 0).when(col("INMUSUPR") == "1", 1).otherwise(np.nan))
df2 = df2.withColumn("HYPERTENSION",when(col("HYPERTENSION") == "2", 0).when(col("HYPERTENSION") == "1", 1).otherwise(np.nan))
df2 = df2.withColumn("OTHER_DISEASE",when(col("OTHER_DISEASE") == "2", 0).when(col("OTHER_DISEASE") == "1", 1).otherwise(np.nan))
df2 = df2.withColumn("CARDIOVASCULAR",when(col("CARDIOVASCULAR") == "2", 0).when(col("CARDIOVASCULAR") == "1", 1).otherwise(np.nan))
df2 = df2.withColumn("OBESITY",when(col("OBESITY") == "2", 0).when(col("OBESITY") == "1", 1).otherwise(np.nan))
df2 = df2.withColumn("RENAL_CRONIC",when(col("RENAL_CRONIC") == "2", 0).when(col("RENAL_CRONIC") == "1", 1).otherwise(np.nan))
df2 = df2.withColumn("TOBACCO",when(col("TOBACCO") == "2", 0).when(col("TOBACCO") == "1", 1).otherwise(np.nan))
#leave out contact_other_covid
df2 = df2.withColumn("CONTACT_OTHER_COVID",when(col("CONTACT_OTHER_COVID") == "2", 0).when(col("CONTACT_OTHER_COVID") == "1", 1).otherwise(np.nan))
df2 = df2.withColumn("ICU",when(col("ICU") == "2", 0).when(col("ICU") == "1", 1).otherwise(np.nan))
df2 = df2.withColumn("DEATH",when(col("DEATH") == "9999-99-99", 0).otherwise("1"))
df2 = df2.withColumn("LAB_RESULT",when(col("LAB_RESULT") == "2", 0).when(col("LAB_RESULT") == "1", 1).otherwise(np.nan))

df2.show(10);


# In[10]:


#converts the nan to unknown
df_unknown_imputed = df2.fillna(0)

df_unknown_imputed.cache()

#number of entries on this dataset
print(df_unknown_imputed.count())
df_unknown_imputed.groupBy("DEATH").count().sort("count",ascending=True).show()


# In[11]:


from pyspark.sql.types import *

df_unknown_imputed.printSchema()
df_unknown_imputed = df_unknown_imputed.withColumn("DEATH", df_unknown_imputed["DEATH"].cast(DoubleType()))

distinctDF = df_unknown_imputed


# In[12]:


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


# In[13]:


#take the caracteristics, and symptoms as the category columns 'features'
catcols = ['SEX','AGE','NEUMONIA', 'PREGNANT','DIABETES', 'EPOC', 'ASTHMA', 'INMUSUPR','HYPERTENSION','OTHER_DISEASE',
           'CARDIOVASCULAR','OBESITY','RENAL_CRONIC','TOBACCO']

#these are results from the symptoms
num_cols = ['DEATH', 'ICU','INTUBED']
# num_cols = ['DEATH', 'ICU']
#try to predict this (outpatient vs inpatient), can predict death and ICU later
labelCol = 'PATIENT_TYPE'

data = get_dummy(distinctDF,catcols,num_cols,labelCol)
data.show(5)


# In[14]:


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


# In[15]:


from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.util import MLUtils

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)


# In[16]:


evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print(accuracy)


# In[17]:


import six
for i in df_unknown_imputed.columns:
    if not( isinstance(df_unknown_imputed.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to PATIENT_TYPE for ", i, df_unknown_imputed.stat.corr('PATIENT_TYPE',i))


# In[18]:


import six
for i in df_unknown_imputed.columns:
    if not( isinstance(df_unknown_imputed.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to ICU for ", i, df_unknown_imputed.stat.corr('ICU',i))


# In[19]:


import six
for i in df_unknown_imputed.columns:
    if not( isinstance(df_unknown_imputed.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to INTUBED for ", i, df_unknown_imputed.stat.corr('INTUBED',i))


# In[3]:


#Produce the HeapMap on entire raw Data
#Observe that symtopms are major contribution

# Step 0 - Read the dataset, calculate column correlations and make a seaborn heatmap
import seaborn as sns

data = df = pd.read_csv('201029COVID19MEXICO.csv',encoding='latin1')

corr = data.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[4]:


df_new = data.rename(columns={'SEXO': 'SEX',
                             'Age' : 'AGE',
                              'INTUBADO' : 'INTUBED',
                        "NEUMONIA": "NEUMONIA",
                        "Pregnant": "PREGNANT",
                       "DIABETES": "DIABETES",
                      "EPOC" : "EPOC",
                       "ASMA" : "ASTHMA",
                       "INMUSUPR":"INMUSUPR",
                       "HIPERTENSION":"HYPERTENSION",
                       "CARDIOVASCULAR":"CARDIOVASCULAR",
                       "OBESIDAD":"OBESITY",
                       "RENAL_CRONICA": "RENAL_CRONIC",
                       "OTRA_COM": "OTHER_DISEASE",
                      "TABAQUISMO":"TOBACCO",
                       "OTRO_CASO": "CONTACT_OTHER_COVID",
                       "CLASIFICACION_FINAL": "CLASSIFICACION_FINAL",
                       "UCI":"ICU",
                       "RESULTADO_LAB":"LAB_RESULT",
                        "FECHA_DEF":"DEATH",
                         'TIPO_PACIENTE': 'PATIENT_TYPE'     
                             })



print(df_new.columns)               


# In[5]:


# Step 1 - Make a scatter plot with square markers, set column names as labels

def heatmap(x, y, size):
    fig, ax = plt.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    
    size_scale = 500
    ax.scatter(
        cmap=sns.diverging_palette(20, 220, n=200),
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    
columns = ['SEX', 'NEUMONIA', 'AGE', 'PREGNANT', 'DIABETES', 'EPOC','ASTHMA','INMUSUPR','HYPERTENSION','OTHER_DISEASE','CARDIOVASCULAR','OBESITY',
          'RENAL_CRONIC','TOBACCO','CONTACT_OTHER_COVID'] 

corr = df_new[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)


# In[6]:


df_new.head(5)


# In[7]:


print(df_new.columns)


# In[8]:


df = df_new.drop(columns=['FECHA_ACTUALIZACION', 'ID_REGISTRO','ORIGEN','SECTOR','ENTIDAD_UM','ENTIDAD_NAC','ENTIDAD_RES','MUNICIPIO_RES','FECHA_INGRESO','FECHA_SINTOMAS','NACIONALIDAD','TOMA_MUESTRA','CLASSIFICACION_FINAL','MIGRANTE','PAIS_NACIONALIDAD','PAIS_ORIGEN','LAB_RESULT','HABLA_LENGUA_INDIG','Native'])

print(df.columns)


# In[9]:


df['PREGNANT'] = df['PREGNANT'].replace([2],0)
df['PREGNANT'] = df['PREGNANT'].replace([98,97],2)

df['NEUMONIA'] = df['NEUMONIA'].replace([2],0)
df['NEUMONIA'] = df['NEUMONIA'].replace([99],2)

df['INTUBED'] = df['INTUBED'].replace([2],0)
df['INTUBED'] = df['INTUBED'].replace([99,97],2)

df['DIABETES'] = df['DIABETES'].replace([2],0)
df['DIABETES'] = df['DIABETES'].replace([98],2)

df['EPOC'] = df['EPOC'].replace([2],0)
df['EPOC'] = df['EPOC'].replace([98],2)


# In[10]:


df['ASTHMA'] = df['ASTHMA'].replace([2],0)
df['ASTHMA'] = df['ASTHMA'].replace([98],2)

df['INMUSUPR'] = df['INMUSUPR'].replace([2],0)
df['INMUSUPR'] = df['INMUSUPR'].replace([98],2)

df['CARDIOVASCULAR'] = df['CARDIOVASCULAR'].replace([2],0)
df['CARDIOVASCULAR'] = df['CARDIOVASCULAR'].replace([98],2)

df['HYPERTENSION'] = df['HYPERTENSION'].replace([2],0)
df['HYPERTENSION'] = df['HYPERTENSION'].replace([98],2)

df['OTHER_DISEASE'] = df['OTHER_DISEASE'].replace([2],0)
df['OTHER_DISEASE'] = df['OTHER_DISEASE'].replace([98],2)


df['OBESITY'] = df['OBESITY'].replace([2],0)
df['OBESITY'] = df['OBESITY'].replace([98],2)

df['RENAL_CRONIC'] = df['RENAL_CRONIC'].replace([2],0)
df['RENAL_CRONIC'] = df['RENAL_CRONIC'].replace([98],2)

df['TOBACCO'] = df['TOBACCO'].replace([2],0)
df['TOBACCO'] = df['TOBACCO'].replace([98],2)

df['CONTACT_OTHER_COVID'] = df['CONTACT_OTHER_COVID'].replace([2],0)
df['CONTACT_OTHER_COVID'] = df['CONTACT_OTHER_COVID'].replace([99],2)


# In[11]:


df['ICU'] = df['ICU'].replace([2],0)
df['ICU'] = df['ICU'].replace([97,98,99],2)

df['DEATH'] = df['DEATH'].replace(['9999-99-99'],0)


# In[12]:


df['SEX'].unique()


# In[14]:


#No male with pregnant = true
df.loc[(df['PREGNANT'] == 1) & (df['SEX'] == 2)]


# In[15]:


#Row/Column values - True=1, Flase-0, Unknown-2
##Male - 2 #Female - 1 : SEX

df.head(5)


# In[16]:


corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

#This heatMap is different from previous since we have removed unwanted columns and refined data


# In[17]:


def heatmap(x, y, size):
    fig, ax = plt.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    
    size_scale = 500
    ax.scatter(
        cmap=sns.diverging_palette(20, 220, n=200),
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    
columns = ['SEX', 'NEUMONIA', 'AGE', 'PREGNANT', 'DIABETES', 'EPOC','ASTHMA','INMUSUPR','HYPERTENSION','OTHER_DISEASE','CARDIOVASCULAR','OBESITY',
          'RENAL_CRONIC','TOBACCO','CONTACT_OTHER_COVID'] 

corr = df[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)


# In[22]:


df1 = df.reindex(['PATIENT_TYPE', 'DEATH', 'INTUBED','ICU','AGE','SEX','PREGNANT', 'DIABETES', 'EPOC', 'ASTHMA', 'INMUSUPR', 'HYPERTENSION',
       'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CRONIC', 'TOBACCO','CONTACT_OTHER_COVID'], axis="columns")
df1.head(1)


# In[34]:


from sklearn.model_selection import train_test_split
Y = df1.iloc[:,:4]

X = df1.iloc[:,4:]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=12)


# In[50]:





# In[ ]:




