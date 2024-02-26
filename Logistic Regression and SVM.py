#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[2]:


dataset = pd.read_csv("emotional.antecedents.csv")


# In[3]:


dataset.head()


# In[4]:


dataset.drop(["ID","CITY","COUN","SUBJ","RELI","PRAC","FOCC","MOCC","FIEL","PLAN","FAIR","CAUS","COPING","MORL","NEUTRO","MYKEY","SIT","STATE"], axis=1, inplace=True)


# In[5]:


dataset.head()


# In[6]:


dataset.shape


# In[7]:


dataset.info()


# In[8]:


x=dataset.drop('Field1', axis=1)
y=dataset['Field1']


# In[9]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[10]:


scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[11]:


clf = LogisticRegression(random_state=0).fit(x_train, y_train)


# In[12]:


y_pred_test=clf.predict(x_test)


# In[13]:


accuracy_score(y_test,y_pred_test)


# In[14]:


print('Test Accuracy: %f'%(np.mean(y_pred_test == y_test) * 100))


# In[15]:


cm = confusion_matrix(y_test,y_pred_test)


# In[16]:


ax = sns.heatmap(cm, cmap="flare",annot=True, fmt="d")

plt.xlabel("Predicted Class",fontsize=12)
plt.ylabel("True Class",fontsize=12)
plt.title("Confusion Matrix",fontsize=12)

plt.show()


# In[17]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))


# In[ ]:


###### implementing SVM model ########


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[2]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[3]:


dataset = pd.read_csv("Humanemotional_antecedents.csv")


# In[4]:


x=dataset.drop('Field1', axis=1)
y=dataset['Field1']


# In[5]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[6]:


scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[7]:


svc=SVC() 
svc.fit(x_train,y_train)

y_pred_test=svc.predict(x_test)

print('Model accuracy : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


# In[8]:


linear_svc=SVC(kernel='linear', C=1.0) 
linear_svc.fit(x_train,y_train)

y_pred_test=linear_svc.predict(x_test)

print('C=1.0 Model accuracy with linear kernel : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


# In[9]:


svc=SVC(C=100.0) 
svc.fit(x_train,y_train)

y_pred_test=svc.predict(x_test)

print('C=100.0 Model accuracy with rbf kernel : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


# In[10]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))


# In[11]:


print('Test Accuracy: %f'%(np.mean(y_pred_test == y_test) * 100))


# In[12]:


cm = confusion_matrix(y_test,y_pred_test)


# In[13]:


ax = sns.heatmap(cm, cmap="flare",annot=True, fmt="d")

plt.xlabel("Predicted Class",fontsize=12)
plt.ylabel("True Class",fontsize=12)
plt.title("Confusion Matrix",fontsize=12)

plt.show()


# In[ ]:




