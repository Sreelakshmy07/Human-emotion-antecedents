#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Required Libraries


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[3]:


dataset = pd.read_csv("emotional.antecedents.csv")


# In[4]:


dataset.shape


# In[5]:


dataset.head()


# In[6]:


dataset.tail()


# In[7]:


dataset.info()


# In[8]:


dataset.drop(['STATE'], axis=1, inplace=True)


# In[9]:


# Categorical variables


# In[10]:


categorical = [var for var in dataset.columns if dataset[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)


# In[11]:


dataset[categorical].head()


# In[12]:


print(dataset["Field1"].value_counts())


# In[13]:


dataset.drop(["ID","CITY","COUN","SUBJ","RELI","PRAC","FOCC","MOCC","FIEL","PLAN","MOVE","TEMPER","FAIR","CAUS","COPING","MORL","NEUTRO","MYKEY","SIT"], axis=1, inplace=True)


# In[14]:


dataset.shape


# In[15]:


dataset.head()


# In[16]:


dataset.info()


# In[17]:


X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]].values
Y = dataset.iloc[:, 21].values


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state =1)


# In[19]:


len(X_train)


# In[20]:


len(X_test)


# In[21]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()


# In[22]:


model.fit(X_train, Y_train)


# In[23]:


import sklearn
from sklearn.metrics import accuracy_score

import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20)
gnb = MultinomialNB()
Y_pred = gnb.fit(X_train, Y_train).predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)


# In[24]:


model.score(X_test, Y_test)


# In[25]:


from sklearn.naive_bayes import MultinomialNB

gnb = MultinomialNB()

gnb.fit(X_train, Y_train)


# In[26]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[27]:


ax = sns.heatmap(cm, cmap="flare",annot=True, fmt="d")

plt.xlabel("Predicted Class",fontsize=12)
plt.ylabel("True Class",fontsize=12)
plt.title("Confusion Matrix",fontsize=12)

plt.show()


# In[28]:


from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred))


# In[29]:


from sklearn.ensemble import BaggingClassifier
base_clf = MultinomialNB(alpha=0.1)
bag_clf = BaggingClassifier(base_estimator=base_clf, n_estimators=50)

# Train the bagging classifier
bag_clf.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = bag_clf.predict(X_test)


# In[30]:


accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[31]:


from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred))

