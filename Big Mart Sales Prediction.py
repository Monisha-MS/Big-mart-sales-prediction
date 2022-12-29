#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


train=pd.read_csv("train_data_path.csv")
test=pd.read_csv("test_data_path.csv")


# In[3]:


train.head()


# In[79]:


train.drop(['Item_Identifier','Outlet_Identifier'],axis=1)


# In[86]:


test.head()


# In[6]:


test.drop(['Item_Identifier','Outlet_Identifier'],axis=1)


# In[7]:





# In[8]:


test.describe()


# In[80]:


train.info()


# In[10]:


test.info()


# In[11]:


train.apply(lambda x:len(x.unique()))


# In[12]:


test.apply(lambda x: len(x.unique()))


# In[88]:


train['Item_Fat_Content'].value_counts()


# In[134]:


#EDA


# In[16]:


from matplotlib import style
print(plt.style.available)


# In[42]:


plt.style.use('classic')
plt.figure(figsize=(20,9))
sns.distplot(train['Item_Outlet_Sales'],bins=20,color='r')


# In[97]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,7))
sns.distplot(train['Item_Weight'],bins=25)


# In[89]:


sns.countplot(train.Item_Fat_Content)


# In[59]:


plt.figure(figsize=(20,10))
sns.distplot(train['Item_Visibility'])
plt.show()


# In[62]:


sns.countplot(train['Item_Type'])
plt.xticks(rotation=90)
plt.show()


# In[65]:


sns.distplot(train['Item_MRP'])


# In[66]:


sns.countplot(train['Outlet_Establishment_Year'])


# In[67]:


sns.countplot(train['Outlet_Size'])


# In[68]:


sns.countplot(train['Outlet_Location_Type'])


# In[69]:


sns.countplot(train['Outlet_Type'])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


#bivariate analysis


# In[103]:


Fat_plot = pd.pivot_table(train,index='Item_Fat_Content', values='Item_Outlet_Sales', aggfunc=np.mean)
Fat_plot.plot(kind='bar', figsize=(8, 8))
plt.xlabel('Item_Fat_Content')
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Fat_Content and Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[102]:


establishment_plot = pd.pivot_table(train,index='Outlet_Establishment_Year', values='Item_Outlet_Sales', aggfunc=np.mean)
establishment_plot.plot(kind='bar', figsize=(8, 8),color='g')
plt.xlabel('Outlet_Establishment_Year')
plt.ylabel("Item_Outlet_Sales")
plt.title("Outlet_Establishment_Year and Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[108]:


type_plot = pd.pivot_table(train,index='Item_Type', values='Item_Outlet_Sales', aggfunc=np.mean)
type_plot.plot(kind='bar', figsize=(8, 8),color='r')
plt.xlabel('Item_Type')
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Type and Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[109]:


size_plot = pd.pivot_table(train,index='Outlet_Size', values='Item_Outlet_Sales', aggfunc=np.mean)
size_plot.plot(kind='bar', figsize=(8, 8),color='g')
plt.xlabel('Outlet_Size')
plt.ylabel("Item_Outlet_Sales")
plt.title("Outlet_Size and Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[107]:


location_plot = pd.pivot_table(train,index='Outlet_Location_Type', values='Item_Outlet_Sales', aggfunc=np.mean)
location_plot.plot(kind='bar', figsize=(8, 8),color='y')
plt.xlabel('Outlet_Location_Type')
plt.ylabel("Item_Outlet_Sales")
plt.title("Outlet_Location_Type and Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[111]:


identifier_plot = pd.pivot_table(train,index='Outlet_Identifier', values='Item_Outlet_Sales', aggfunc=np.mean)
identifier_plot.plot(kind='bar', figsize=(8, 8))
plt.xlabel('Outlet_Identifier')
plt.ylabel("Item_Outlet_Sales")
plt.title("Outlet_Identifier and Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[71]:


#data preprocessing
train.isnull().sum()


# In[74]:


train['Item_Weight']=train['Item_Weight'].fillna(-2).astype('float64')
train['Outlet_Size']=train['Outlet_Size'].fillna(-2).astype('object')


# In[75]:


train.isnull().sum()


# In[76]:


test.isnull().sum()


# In[77]:


test['Item_Weight']=test['Item_Weight'].fillna(-2).astype('float64')
test['Outlet_Size']=test['Outlet_Size'].fillna(-2).astype('object')


# In[78]:


test.isnull().sum()


# In[ ]:


#categorical values


# In[125]:


data=pd.concat([train,test],ignore_index=True)
data.head()


# In[126]:


cols=['Item_Identifier','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in cols:
    data[col]=le.fit_transform(data[col])
data.head()


# In[121]:





# In[112]:


cols=['Item_Fat_Content','Item_Type','Outlet_Size']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in cols:
    train[col]=le.fit_transform(train[col])
train.head()


# In[114]:


print('Original Categories:')
print(train['Item_Fat_Content'].value_counts())






