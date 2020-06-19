#!/usr/bin/env python
# coding: utf-8

# In[ ]:


after opening the dataset , i checked dataset if the first step preprocessing can be applied. and there was numerical and cateorical data asw well 
and some missing values were also there in mulitple rows. after preprocessing , accroding to requirements of project by datacamp, done the project


# In[37]:


import pandas as pd


# In[2]:


df = pd.read_csv('crx.data', header=None)


# ### Credit card applications

# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.isnull().sum().sum()


# In[7]:


description = df.describe()
description


# In[8]:


df.shape


# In[9]:


df.tail(50)


# In[10]:


df.isnull().values.sum()


# In[11]:


import numpy as np


# In[ ]:





# In[12]:


df.tail(50)


# In[13]:


import numpy as np


# In[14]:


df = df.replace('?', np.nan)


# In[15]:


df.tail(50)


# In[16]:


df.fillna(df.mean(), inplace=True)


# In[17]:


df.head()


# In[18]:


df.tail(50)


# In[19]:


print(df.isnull().values.sum())


# In[20]:


for col in df:
   if df[col].dtypes == 'object':
        df= df.fillna(df[col].value_counts().index[0])
        
    


# In[21]:


print(df.isnull().values.sum())


# In[22]:


# As few featuress of dataset still seems in text form , so before aplying any model i have to convet it into numerical form.
# to convert it into nuerical form , label encoder is best way to do 


# In[23]:


from sklearn.preprocessing import LabelEncoder


# In[24]:


le = LabelEncoder()


# In[25]:


for col in df.columns:
    if df[col].dtypes=='object':
        df[col]=le.fit_transform(df[col])
        


# In[26]:


df.info()


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


df.head()


# In[29]:


# Import train_test_split
from sklearn.model_selection import train_test_split

# Drop the features 11 and 13 and convert the DataFrame to a NumPy array
df = df.drop([df.columns[11], df.columns[13]], axis=1)
df = df.values

# Segregate features and labels into separate variables
X,y = df[:,0:13] , df[:,13]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size=0.33,
                                random_state=42)


# In[30]:


from sklearn.preprocessing import MinMaxScaler


# In[31]:


scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)


# In[32]:


from sklearn.linear_model import LogisticRegression


# In[33]:


logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(rescaledX_train, y_train)


# In[34]:


from sklearn.metrics import confusion_matrix

y_pred = logreg.predict(rescaledX_test)

print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))

confusion_matrix(y_pred, y_test)


# In[35]:


from sklearn.model_selection import GridSearchCV
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

param_grid = dict(tol=tol, max_iter=max_iter)
print(param_grid)


# In[36]:


grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

rescaledX = scaler.fit_transform(X)

grid_model_result = grid_model.fit(rescaledX, y)

best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))


# In[ ]:




