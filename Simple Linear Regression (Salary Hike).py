#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


sal_hike=pd.read_csv("C://Users//hp//Downloads//Salary_Data.csv")


# In[5]:


plt.scatter(sal_hike.YearsExperience,sal_hike.Salary)
np.corrcoef(sal_hike.YearsExperience,sal_hike.Salary)


# In[6]:


#r=0.978
sal_hike.corr()


# In[7]:


#Linear Model


# In[8]:


import statsmodels.formula.api as smf
lin_model=smf.ols('sal_hike.Salary~sal_hike.YearsExperience',data=sal_hike).fit()


# In[9]:


lin_model.params
lin_model.summary()


# In[10]:


#will check for p-value<0.05 and high R-squared value(0.957) to be a good model,if not go for transformation
lin_predict=lin_model.predict(sal_hike)
lin_predict


# In[11]:


lin_Error=sal_hike.Salary-lin_predict
lin_Error


# In[12]:


plt.scatter(sal_hike.YearsExperience,sal_hike.Salary,c='r');plt.plot(sal_hike.YearsExperience,lin_predict,c='b');plt.xlabel('years of experience');plt.ylabel('salary')
lin_model.conf_int()


# In[13]:


np.corrcoef(sal_hike.Salary,lin_predict)
#r=0.978


# In[14]:


from sklearn.metrics import mean_squared_error
from math import sqrt
lin_rmse=sqrt(mean_squared_error(sal_hike.Salary,lin_predict)) 
lin_rmse
#rmse=5592, R_sq=0.957


# In[15]:


#Log Model


# In[16]:


import statsmodels.formula.api as smf
log_model=smf.ols('sal_hike.Salary~np.log(sal_hike.YearsExperience)',data=sal_hike).fit()
log_model.params
log_model.summary()


# In[17]:


#will check for p-value<0.05 and high R-squared value(0.854) to be a good model,if not go for transformation
log_predict=log_model.predict(sal_hike)
log_predict


# In[18]:


log_Error=sal_hike.Salary-log_predict
log_Error


# In[19]:


plt.scatter(sal_hike.YearsExperience,sal_hike.Salary,c='r');plt.plot(sal_hike.YearsExperience,log_predict,c='b');plt.xlabel('years of experience');plt.ylabel('salary')
log_model.conf_int()


# In[20]:


np.corrcoef(sal_hike.Salary,log_predict)
#r=0.924


# In[21]:


from sklearn.metrics import mean_squared_error
from math import sqrt
log_rmse=sqrt(mean_squared_error(sal_hike.Salary,log_predict)) 
log_rmse
#rmse=10302.89, R_sq=0.854


# In[22]:


#Exponential Model


# In[23]:


import statsmodels.formula.api as smf
Exp_model=smf.ols('np.log(sal_hike.Salary)~sal_hike.YearsExperience',data=sal_hike).fit()
Exp_model.params
Exp_model.summary()


# In[24]:


#will check for p-value<0.05 and high R-squared value(0.932) to be a good model,if not go for transformation
pred=Exp_model.predict(sal_hike)
Exp_predict=np.exp(pred)
Exp_predict


# In[25]:


Exp_Error=sal_hike.Salary-Exp_predict
Exp_Error


# In[26]:


plt.scatter(sal_hike.YearsExperience,sal_hike.Salary,c='r');plt.plot(sal_hike.YearsExperience,Exp_predict,c='b');plt.xlabel('years of experience');plt.ylabel('salary')
Exp_model.conf_int()


# In[27]:


np.corrcoef(sal_hike.Salary,Exp_predict)
#r=0.966


# In[29]:


from sklearn.metrics import mean_squared_error
from math import sqrt
Exp_rmse=sqrt(mean_squared_error(sal_hike.Salary,Exp_predict)) 
Exp_rmse
#rmse=7213.23, R_sq=0.932


# In[30]:


#Quad Model


# In[31]:


import statsmodels.formula.api as smf
#sal_hike['sq_exp']=sal_hike.YearsExperience*sal_hike.YearsExperience
#sal_hike.drop('sq_exp',axis=1,inplace=True)
#sal_hike


# In[32]:


Quad_model=smf.ols('sal_hike.Salary~(sal_hike.YearsExperience*sal_hike.YearsExperience+sal_hike.YearsExperience)',data=sal_hike).fit()
#Quad_model=smf.ols('sal_hike.Salary~sal_hike.sq_exp+sal_hike.YearsExperience',data=sal_hike).fit()
Quad_model.params
Quad_model.summary()


# In[33]:


#will check for p-value<0.05 and high R-squared value(0.957) to be a good model,if not go for transformation
Quad_predict=Quad_model.predict(sal_hike)
Quad_predict


# In[34]:


Quad_Error=sal_hike.Salary-Quad_predict
Quad_Error


# In[35]:


plt.scatter(sal_hike.YearsExperience,sal_hike.Salary,c='r');plt.plot(sal_hike.YearsExperience,Quad_predict,c='b');plt.xlabel('years of experience');plt.ylabel('salary')
Quad_model.conf_int()


# In[36]:


np.corrcoef(sal_hike.Salary,Quad_predict)
#r=0.978


# In[37]:


from sklearn.metrics import mean_squared_error
from math import sqrt


# In[39]:


Quad_rmse=sqrt(mean_squared_error(sal_hike.Salary,Quad_predict)) 
Quad_rmse
#rmse=5592.04, R_sq=0.957


# In[ ]:




