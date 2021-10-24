#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


bank=pd.read_csv('C:\\Users\\hp\\Downloads\\bank-full.csv',sep=';')


# In[3]:


bank.columns
bank.isna().sum()


# In[4]:


#Model building
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
bank['job']=label_encoder.fit_transform(bank['job'])
bank['marital']=label_encoder.fit_transform(bank['marital'])
bank['education']=label_encoder.fit_transform(bank['education'])
bank['default']=label_encoder.fit_transform(bank['default'])
bank['housing']=label_encoder.fit_transform(bank['housing'])
bank['loan']=label_encoder.fit_transform(bank['loan'])
bank['contact']=label_encoder.fit_transform(bank['contact'])
bank['month']=label_encoder.fit_transform(bank['month'])
bank['poutcome']=label_encoder.fit_transform(bank['poutcome'])
bank['y']=label_encoder.fit_transform(bank['y'])


# In[5]:


import statsmodels.formula.api as smf
bank_model=smf.logit('y~age+job+marital+education+default+balance+housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome',data=bank).fit()
bank_model.summary()


# In[6]:


#job is insignificant
bank['job_sq']=bank['job']*bank['job']
bank.drop(['job_sq'],axis=1,inplace=True)
bank_model1=smf.logit('y~age+marital+education+default+balance+housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome',data=bank).fit()
bank_model1.summary()
bank_pred=bank_model1.predict(bank)
bank_pred
bank['y_pred']=0
bank.loc[bank_pred>=0.5,'y_pred']=1
bank.y_pred


# In[7]:


#confusion matrix
from sklearn.metrics import classification_report
classification_report(bank['y'],bank['y_pred'])
confusion_matrix=pd.crosstab(bank.y,bank.y_pred)
confusion_matrix
accuracy=(39139+1137)/(39139+783+4152+1137)
accuracy


# In[9]:


#ROC curve
from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(bank.y,bank_pred)
plt.plot(fpr,tpr);plt.xlabel('false positive');plt.ylabel('true positive')
roc_auc=metrics.auc(fpr,tpr)#area under curve
roc_auc


# In[ ]:




