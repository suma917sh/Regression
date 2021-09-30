#!/usr/bin/env python
# coding: utf-8

# In[1]:


# To find out Delivery Prediction


# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
delv_time=pd.read_csv("C://Users//hp//Downloads//delivery_time1.csv")


# In[8]:


plt.scatter(delv_time.Sorting_time,delv_time.Delivery_time)
np.corrcoef(delv_time.Sorting_time,delv_time.Delivery_time)
#r=0.825


# In[11]:


#Linear Model
import statsmodels.formula.api as smf
lin_model=smf.ols('delv_time.Delivery_time~delv_time.Sorting_time',data=delv_time).fit()
lin_model.params
lin_model.summary()


# In[12]:


##will check for p-value<0.05 and high R-squared value(0.682) to be a good model,if not go for transformation
lin_predict=lin_model.predict(delv_time)
lin_predict


# In[13]:


lin_error=delv_time.Delivery_time-lin_predict
lin_error


# In[14]:


plt.scatter(x=delv_time.Sorting_time,y=delv_time.Delivery_time,c='r');plt.plot(delv_time.Sorting_time,lin_predict,c='b');plt.xlabel('sorting time');plt.ylabel('delivery time');
lin_model.conf_int(0.05)
lin_corr=np.corrcoef(delv_time.Delivery_time,lin_predict)
lin_corr
#r=0.825


# In[15]:


from sklearn.metrics import mean_squared_error
from math import sqrt
lin_rmse=sqrt(mean_squared_error(delv_time.Delivery_time,lin_predict))
lin_rmse
#rmse=2.79


# In[17]:


#Log Model
import statsmodels.formula.api as smf
log_model=smf.ols('delv_time.Delivery_time~np.log(delv_time.Sorting_time)',data=delv_time).fit()
log_model.params
log_model.summary()


# In[18]:


##will check for p-value<0.05 and high R-squared value(0.695) to be a good model,if not go for transformation
log_predict=log_model.predict(delv_time)
log_predict


# In[19]:


log_error=delv_time.Delivery_time-log_predict
log_error


# In[20]:


plt.scatter(x=delv_time.Sorting_time,y=delv_time.Delivery_time,c='r');plt.plot(delv_time.Sorting_time,log_predict,c='b');plt.xlabel('sorting time');plt.ylabel('delivery time');
log_model.conf_int(0.05)
log_corr=np.corrcoef(delv_time.Delivery_time,log_predict)
log_corr
#r=0.833


# In[21]:


from sklearn.metrics import mean_squared_error
from math import sqrt
log_rmse=sqrt(mean_squared_error(delv_time.Delivery_time,log_predict))
log_rmse
#rmse=2.733


# In[22]:


#Exponential Model
import statsmodels.formula.api as smf
exp_model=smf.ols('np.log(delv_time.Delivery_time)~delv_time.Sorting_time',data=delv_time).fit()
exp_model.params
exp_model.summary()


# In[23]:


##will check for p-value<0.05 and high R-squared value(0.711) to be a good model,if not go for transformation
predi=exp_model.predict(delv_time)
exp_predict=np.exp(predi)
exp_predict


# In[24]:


exp_error=delv_time.Delivery_time-exp_predict
exp_error


# In[25]:


plt.scatter(x=delv_time.Sorting_time,y=delv_time.Delivery_time,c='r');plt.plot(delv_time.Sorting_time,exp_predict,c='b');plt.xlabel('sorting time');plt.ylabel('delivery time');
exp_model.conf_int(0.05)
exp_corr=np.corrcoef(delv_time.Delivery_time,exp_predict)
exp_corr
#r=0.808


# In[26]:


from sklearn.metrics import mean_squared_error
from math import sqrt
exp_rmse=sqrt(mean_squared_error(delv_time.Delivery_time,exp_predict))
exp_rmse
#rmse=2.94


# In[28]:


#Quadratic Model
import statsmodels.formula.api as smf
delv_time['sq_sort_time']=delv_time.Sorting_time*delv_time.Sorting_time
Quad_model=smf.ols('delv_time.Delivery_time~delv_time.sq_sort_time+delv_time.Sorting_time',data=delv_time).fit()
Quad_model.params
Quad_model.summary()


# In[30]:


##will check for p-value<0.05 and high R-squared value(0.693) to be a good model,if not go for transformation
Quad_predict=Quad_model.predict(delv_time)
Quad_predict


# In[31]:


Quad_error=delv_time.Delivery_time-Quad_predict
Quad_error


# In[32]:


plt.scatter(x=delv_time.Sorting_time,y=delv_time.Delivery_time,c='r');plt.plot(delv_time.Sorting_time,Quad_predict,c='b');plt.xlabel('sorting time');plt.ylabel('delivery time');
Quad_model.conf_int(0.05)
Quad_corr=np.corrcoef(delv_time.Delivery_time,Quad_predict)
Quad_corr
#r=0.832


# In[33]:


from sklearn.metrics import mean_squared_error
from math import sqrt
Quad_rmse=sqrt(mean_squared_error(delv_time.Delivery_time,Quad_predict))
Quad_rmse
#rmse=2.74


# In[ ]:




