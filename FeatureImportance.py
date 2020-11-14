#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Raw Data Training and Testing , See their accuracy and Feature Importance
#Random Forest Classifier
#Random Forest Regressor
#Logistic Regression 


# In[2]:


#Random Forest feature importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
import pandas as pd 


df = pd.read_csv('201029COVID19MEXICO.csv',encoding='latin1')
df.head(4)

#Y = distinctDF.filter(['SEX','AGE','NEUMONIA', 'PREGNANT','DIABETES', 'EPOC', 'ASTHMA', 'INMUSUPR','HYPERTENSION','OTHER_DISEASE','CARDIOVASCULAR','OBESITY','RENAL_CRONIC','TOBACCO'])
#Y = distinctDF.selectExpr('SEX','AGE','NEUMONIA', 'PREGNANT','DIABETES', 'EPOC', 'ASTHMA', 'INMUSUPR',
                          #'HYPERTENSION','OTHER_DISEASE','CARDIOVASCULAR','OBESITY','RENAL_CRONIC','TOBACCO')
#Y = df.iloc[:, -1].values
#print(Y)

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=12)


# In[3]:


#patient type is #9
Y = df.iloc[:,9]
print(Y)

X = df[['SEXO', 'Age','NEUMONIA','Pregnant','DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA','OTRA_COM','TABAQUISMO']]
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=12)


# In[4]:


rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)


# In[5]:


accuracy = rf.score(X_train,y_train)
print( 'Random Forest Regression Accuracy: ', accuracy*100,'%')


# In[6]:


import numpy as np 
import seaborn as sns

def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


# In[7]:


plot_feature_importance(rf.feature_importances_,X.columns,'RANDOM FOREST REGRESSOR')


# In[8]:


# Import needed packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# Make predictions for the test set
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
y_pred_test = forest.predict(X_test)
accuracy_score(y_test, y_pred_test)


# In[9]:


print(classification_report(y_test, y_pred_test))


# In[10]:


# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['SEXO', 'Age','NEUMONIA','Pregnant','DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA','OTRA_COM','TABAQUISMO']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()


# In[11]:


plot_feature_importance(forest.feature_importances_,X.columns,'RANDOM FOREST CLASSIFIER')


# In[12]:


#logistic Regression
from sklearn.linear_model import LogisticRegression


logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train)


# In[14]:


y_pred = logistic_regression.predict(X_test)
accuracy = logistic_regression.accuracy_score(y_test, y_pred)
accuracy_percentage = 100 * accuracy
accuracy_percentage


# In[ ]:


from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import matplotlib.pyplot as plt

# get importance
importance = logistic_regression.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[ ]:


#HeapMap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


# In[ ]:


label_encoder = LabelEncoder()
X.iloc[:,0] = label_encoder.fit_transform(X.iloc[:,0]).astype('float64')


corr = X.corr()


# In[ ]:




