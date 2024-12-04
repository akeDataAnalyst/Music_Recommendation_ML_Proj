#!/usr/bin/env python
# coding: utf-8

# ### Recommending music albums based on user profiles (age and gender)
# - The model will learn patterns from existing user data to make predictions for new users

# ### 1. import data

# In[1]:


import pandas as pd
music_data= pd.read_csv('music.csv')
music_data.head(2)


# ### 2. CLean or prepare data

# - Creating Input and Output Sets

# In[2]:


x=music_data.drop(columns=['genre'])
x.head(5)   #inpute


# In[3]:


y=music_data['genre']
y.head(5)  #outpute


# ### 3.Create model using algorithm | learning and predicting

# - A decision tree algorithm is used to create a model 
# - Import the necessary class from the scikit-learn library
# - This allows the model to learn patterns from the data 

# In[4]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x,y)
predictions=model.predict([ [21,1],[22,0] ])
predictions


# -  After training, the model can make predictions based on new input data. 
# -  For instance, predicting the genre for a 21-year-old male. The model predicts that a 21-year-old male likes Hip Hop

# ### 4. measure the accuracy of model

# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2) # alocating 20% of data for test

model.fit(x_train,y_train)
predictions=model.predict(x_test)

score = accuracy_score(y_test, predictions)
score


# ### 5. Model persistance

# * To avoid retraining the model every time, save it using joblib
# * Load the model for future predictions without retraining

# In[6]:


get_ipython().system('pip install joblib')


# In[7]:


import joblib
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x,y)
joblib.dump(model,'music-reccomender.joblib')


# In[8]:


model=joblib.load('music-reccomender.joblib')
predictions=model.predict([ [21,1] ])
predictions


# ### 6. visualize decision tree

# In[9]:


from sklearn import tree

tree.export_graphviz(model, out_file='music-recommender.dot',
                    feature_names=['age','gender'],
                    class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)

