#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


# In[5]:


spam = pd.read_csv('spam (1).csv')
spam


# In[6]:


spam.shape


# In[7]:


spam.describe()


# In[8]:


spam.info()


# In[9]:


spam.groupby(spam['v1']).size()


# In[10]:


spam.Category = spam.v1.apply(lambda x: 1 if x == 'spam' else 0) 


# In[11]:


spam.head()


# In[13]:


v2 = spam.iloc[:,1] # Messages column
v2.head()


# In[14]:


ifSpam = spam.iloc[:,0] # Spam column
ifSpam.head()


# In[15]:


v2_train, v2_test, ifSpam_train, ifSpam_test = train_test_split(v2, ifSpam, test_size=0.25)


# #### We will use 75% of our dataset for training
# 

# In[17]:


cv = CountVectorizer()


# With CountVectorizer, text is analyzed and word counts are made and these are converted into vectors.
# 
# 

# In[18]:


features = cv.fit_transform(v2_train)


# In[19]:


features_test = cv.transform(v2_test)


# ## Learning and Predicts
# 

# In[20]:


knModel = KNeighborsClassifier(n_neighbors=1)


# In[21]:


knModel.fit(features, ifSpam_train)


# In[22]:


knPredict = knModel.predict(features_test)


# In[23]:


dtModel = tree.DecisionTreeClassifier()


# In[24]:


dtModel.fit(features, ifSpam_train)


# In[25]:


dtPredict = dtModel.predict(features_test)


# In[26]:


svModel = svm.SVC()


# In[27]:


svModel.fit(features,ifSpam_train)


# In[28]:


svPredict = svModel.predict(features_test)


# In[29]:


rfModel = RandomForestClassifier() 


# In[30]:


rfModel.fit(features, ifSpam_train)


# In[31]:


rfPredict = rfModel.predict(features_test)


# ## Visualization

# In[32]:


from sklearn.metrics import plot_confusion_matrix,plot_precision_recall_curve,plot_roc_curve


# In[33]:


def visualization(model):
    predict = model.predict(features_test)
    plot_confusion_matrix(model,features_test,ifSpam_test)
    plot_precision_recall_curve(model,features_test,ifSpam_test)
    plot_roc_curve(model,features_test,ifSpam_test)


# ### Support Vector Machine
# 

# In[34]:


print("Number of mislabeled out of a total of %d test entries: %d" % (features_test.shape[0], 
                                                                      (ifSpam_test != svPredict).sum()))


# In[35]:


successRate = 100.0 * f1_score(ifSpam_test, svPredict, average='micro')


# In[36]:


print("The Success Rate was calculated as % : " + str(successRate) + " with Support Vector Machine")


# In[37]:


visualization(svModel)## Support vector model


# In[38]:


visualization(dtModel)## Decision Tree Model


# In[39]:


visualization(rfModel)##Random forest model


# In[ ]:





# In[ ]:




