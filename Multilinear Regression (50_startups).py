#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


#read the data
startups=pd.read_csv("C://Users//hp//Downloads//50_Startups.csv")


# In[6]:


state=pd.get_dummies(startups.State)
start_up=startups.drop('State',axis=1)
start_up.columns
start_up=pd.concat([start_up,state],axis=1)


# In[7]:


start_up.columns=['RD','admin','marketing','profit','california','florida','newyork']
start_up
start_up.isna().sum()
start_up.dropna()


# In[11]:


#EDA
import seaborn as sns
sns.pairplot(start_up)
start_up.corr()


# In[12]:


#Model Building
import statsmodels.formula.api as smf
model=smf.ols('profit~RD+admin+marketing+california+florida+newyork',data=start_up).fit()
model.params
model.summary()
#R_sq=0.951
#admin,marketing are insignificant


# In[ ]:




