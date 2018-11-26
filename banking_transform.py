#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# In[2]:


df = pd.read_csv('data/banking/bank-additional-full.csv', sep=';')

columns = ["job",
           "marital",
           "education",
           "default", "housing",
           "loan",
           "contact",
           "month",
           "day_of_week",
           "poutcome"]


# In[3]:


Numeric_columns=["age",
                 "duration",
                 "campaign",
                 "pdays",
                 "previous",
                 "emp.var.rate",
                 "cons.price.idx",
                 "cons.conf.idx",
                 "euribor3m",
                 "nr.employed"]


# In[4]:


#Label encode
classes={}
for column in columns:
    labelenc=LabelEncoder()
    df[column] = labelenc.fit_transform(df[column])
    classes[column]=labelenc.classes_


# In[5]:


labelenc=LabelEncoder()
df["y"] = labelenc.fit_transform(df["y"])
Numeric_columns+=["y"]


# In[6]:


#Create the new columns name
columns_name_cat=[]
for j in range(len(columns)):
    for i in classes[columns[j]]:
        columns_name_cat.append(columns[j]+"."+i)


# In[7]:


databank_Categorical_onehot=OneHotEncoder().fit_transform(np.array(df[columns])).toarray()


# In[8]:


df=np.append(databank_Categorical_onehot,np.array(df[Numeric_columns]),axis=1)


# In[9]:


df=pd.DataFrame(df,columns=np.append(columns_name_cat,Numeric_columns))


# In[10]:


# This column should be dropped for a realistic predictive model:
# Please read Attribute Information
# on this page:https://archive.ics.uci.edu/ml/datasets/bank+marketing
df=df.drop("duration", axis=1)


# In[11]:


train, test = train_test_split(df, test_size=0.2)


# In[12]:


train.to_csv("data/banking/bank-additional-full-transformed-train.csv", sep=";", index=False)
test.to_csv("data/banking/bank-additional-full-transformed-test.csv", sep=";", index=False)

