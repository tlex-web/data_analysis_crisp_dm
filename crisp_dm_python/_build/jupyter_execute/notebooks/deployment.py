#!/usr/bin/env python
# coding: utf-8

# In[1]:


from helpers.helper_functions import pd, np, msno, go, plt, sns, px, tf


# In[ ]:


with open('./data/random_forest_params.json','r') as file:
    rf_hyper_params = json.load(file)


# In[17]:



import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/test.csv', sep='[,$]' , decimal=".", engine='python')


df.columns = df.columns.str.lower()

