
# coding: utf-8

# In[1]:


import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


train = pd.read_csv("../Titanic/Data/train.csv")
test = pd.read_csv("../Titanic/Data/test.csv")


# In[11]:


train_test = train.sample(frac = 0.1, replace = False, random_state = 2333)
train_train = train.drop(train_test.index)
print(train_test.shape)
print(train_train.shape)
print(train.shape)


# In[63]:


# Linear Regression
import sklearn
from sklearn import linear_model
reg = linear_model.LinearRegression()
train_train_x = train_train.drop(columns = ['Ticket', 'Cabin', 'Survived', 'Name', 'Sex', 'Embarked', 'Age'])
train_train_y = train_train.Survived
reg.fit(train_train_x, train_train_y)


# In[66]:


# fit
train_train_pred = np.array(reg.predict(train_train_x) >= 0.5) * 1
sklearn.metrics.confusion_matrix(train_train_y, train_train_pred, labels=None, sample_weight=None)

# predict
train_test_x = train_test.drop(columns = ['Ticket', 'Cabin', 'Survived', 'Name', 'Sex', 'Embarked', 'Age'])
train_test_y = train_test.Survived
train_test_pred = np.array(reg.predict(train_test_x) >= 0.5) * 1
sklearn.metrics.confusion_matrix(train_test_y, train_test_pred, labels=None, sample_weight=None)

