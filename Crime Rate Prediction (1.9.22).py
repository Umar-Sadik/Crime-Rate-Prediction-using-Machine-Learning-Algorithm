#!/usr/bin/env python
# coding: utf-8

# # Crime Rate Analysis and Pediction

# In[1]:


import pandas as pd
import numpy as np
import time


# In[2]:


start_time = time.time()
df = pd.read_csv('cc_2017.csv')
print("--- %s seconds ---" % (time.time() - start_time))


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.loc[df['Date'].str.contains('01/'), 'T_month'] = 1
df.loc[df['Date'].str.contains('02/'), 'T_month'] = 2
df.loc[df['Date'].str.contains('03/'), 'T_month'] = 3
df.loc[df['Date'].str.contains('04/'), 'T_month'] = 4
df.loc[df['Date'].str.contains('05/'), 'T_month'] = 5
df.loc[df['Date'].str.contains('06/'), 'T_month'] = 6
df.loc[df['Date'].str.contains('07/'), 'T_month'] = 7
df.loc[df['Date'].str.contains('08/'), 'T_month'] = 8
df.loc[df['Date'].str.contains('09/'), 'T_month'] = 9
df.loc[df['Date'].str.contains('10/'), 'T_month'] = 10
df.loc[df['Date'].str.contains('11/'), 'T_month'] = 11
df.loc[df['Date'].str.contains('12/'), 'T_month'] = 12


# In[6]:


df['ID'] = 1


# In[7]:


DS=df
DS['Beat']


# In[ ]:





# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


corrmat = df.corr()
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(corrmat,annot=True)


# In[10]:


plt.figure(figsize = (30, 10))
sns.countplot(y= 'Description', data = df, order = df['Description'].value_counts().iloc[:30].index)


# In[11]:


plt.figure(figsize = (12, 10))
sns.countplot(y= 'T_month', data = df, order = df['T_month'].value_counts().iloc[:12].index)


# In[ ]:





# In[12]:


countpt= df['Primary Type'].value_counts()
countpt


# In[ ]:





# In[ ]:





# In[13]:


from sklearn.preprocessing import LabelEncoder


# In[14]:


le= LabelEncoder()


# In[15]:


label_PrimaryType=le.fit_transform(df['Primary Type'])


# In[16]:


array_PrimaryType=le.classes_
array_PrimaryType


# In[17]:


df['Primary Type']=label_PrimaryType
df['Primary Type']


# In[18]:


array_PrimaryType = pd.DataFrame(array_PrimaryType)
array_PrimaryType


# In[ ]:





# In[19]:


label_Description=le.fit_transform(df['Description'])


# In[20]:


array_Description=le.classes_
array_Description


# In[21]:


df['Description']=label_Description


# In[22]:


array_Description = pd.DataFrame(array_Description)
array_Description


# In[ ]:





# In[23]:


label_LocationDescription=le.fit_transform(df['Location Description'])


# In[24]:


array_LocationDescription=le.classes_
array_LocationDescription


# In[25]:


df['Location Description']=label_LocationDescription


# In[26]:


array_LocationDescription = pd.DataFrame(array_LocationDescription)
array_LocationDescription.head(50)


# In[ ]:





# In[ ]:





# In[27]:


label_Arrest=le.fit_transform(df['Arrest'])


# In[28]:


array_Arrest=le.classes_
array_Arrest


# In[29]:


df['Arrest']=label_Arrest
array_Arrest = pd.DataFrame(array_Arrest)
array_Arrest.head(50)


# In[ ]:





# In[30]:


label_Beat=le.fit_transform(df['Beat'])


# In[31]:


array_Beat=le.classes_
array_Beat


# In[32]:


df['Beat']=label_Beat
array_Beat = pd.DataFrame(array_Beat)
array_Beat


# In[ ]:





# In[33]:


df['Count'] = df.groupby(['Year', 'T_month','Primary Type','Description','Location Description','Beat'])['ID'].transform("count")


# In[34]:


df['Total Arrest'] = df.groupby(['Year', 'T_month','Primary Type','Description','Location Description','Beat','Arrest'])['ID'].transform("count")


# In[ ]:





# In[35]:


df.isnull().sum()


# In[36]:


df= df.fillna(value=0)


# In[37]:


df1=df[['Year', 'T_month', 'Beat', 'Primary Type','Location Description','Count','Total Arrest']]


# In[38]:


df1.isnull().sum()


# In[39]:


df1


# In[ ]:





# In[ ]:





# In[ ]:





# # in a certain YEAR, in a certain MONTH, in a certain LOCATION, particular TYPES of crime's TOTAL NUMBER prediction
# 
# **What will be the total number of Crime??

# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


train= df1.drop(['Count','Total Arrest'],axis=1)
test= df1['Count']


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.20, random_state=5)


# In[43]:


train.dtypes


# In[ ]:





# In[44]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[45]:


bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X_train, y_train)


# In[46]:


dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_train.columns)


# In[47]:


featureScores1 = pd.concat([dfcolumns,dfscores],axis=1)
featureScores1.columns = ['Specs','Score']


# In[48]:


featureScores1


# In[ ]:





# In[ ]:





# In[49]:


from sklearn.neighbors import KNeighborsRegressor


# In[50]:


RegKN1 = KNeighborsRegressor()


# In[51]:


RegKN1.fit(X_train, y_train)


# In[52]:


predKNR1= RegKN1.predict(X_test)


# In[53]:


predKNR1=predKNR1.astype(int)
predKNR1


# In[54]:


SKN1=RegKN1.score(X_test, y_test)


# In[55]:


SKN1= SKN1*100
SKN1


# In[56]:


predKNR1= predKNR1.reshape(-1,1)
y_test= y_test.values.reshape(-1,1)
ConKN1= np.concatenate((y_test,predKNR1),axis=1)


# In[57]:


dataframeKNR1= pd.DataFrame(ConKN1,columns=['y_test','Prediction by KNR'])


# In[58]:


dataframeKNR1


# In[59]:


dfKNR1= np.concatenate((X_test,dataframeKNR1),axis=1)


# In[60]:


dfKNR1= pd.DataFrame(dfKNR1,columns=['Year', 'T_month', 'Beat', 'Primary Type','Location Description','Count','Prediction by KNR'])


# In[61]:


dfKNR1.head(25)


# In[ ]:





# In[ ]:





# In[62]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# In[63]:


RegRF1= RandomForestRegressor(n_estimators= 40, random_state=1)
RegRF1.fit(X_train, y_train)


# In[64]:


predRF1= RegRF1.predict(X_test)


# In[65]:


predRF1=predRF1.astype(int)
predRF1


# In[66]:


SRF1=RegRF1.score(X_test, y_test)


# In[67]:


SRF1= SRF1*100
SRF1


# In[68]:


predRF1= predRF1.reshape(-1,1)
y_test= y_test.reshape(-1,1)
ConRF1= np.concatenate((y_test,predRF1),axis=1)


# In[69]:


dfRF1= pd.DataFrame(ConRF1,columns=['y_test','Prediction by RFR'])


# In[70]:


dfRF1.head(20)


# In[ ]:





# In[71]:


RegDT1= DecisionTreeRegressor(random_state=1, max_features=30)
RegDT1.fit(X_train, y_train)


# In[72]:


predDT1= RegDT1.predict(X_test)


# In[73]:


predDT1=predDT1.astype(int)


# In[74]:


SDT1=RegDT1.score(X_test, y_test)
SDT1=SDT1*100
SDT1


# In[75]:


predDT1= predDT1.reshape(-1,1)
y_test= y_test.reshape(-1,1)
ConDT1= np.concatenate((y_test,predDT1),axis=1)


# In[76]:


dfDT1= pd.DataFrame(ConDT1,columns=['y_test','Prediction by DTR'])


# In[77]:


dfDT1.head(20)


# In[ ]:





# In[ ]:





# In[ ]:





# # in a certain YEAR, in a certain MONTH, in a certain LOCATION, particular TYPES of crime's TOTAL ARREST NUMBER prediction
# 
# **How many will be arrested??

# In[ ]:





# In[78]:


df1


# In[79]:


train2= df1.drop(['Total Arrest'],axis=1)
test2= df1['Total Arrest']


# In[80]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(train2, test2, test_size=0.25, random_state=1)


# In[81]:


bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X_train2, y_train2)


# In[82]:


dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_train2.columns)


# In[83]:


featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']


# In[84]:


featureScores


# In[ ]:





# In[85]:


RegKNR2 = KNeighborsRegressor()


# In[86]:


RegKNR2.fit(X_train2, y_train2)


# In[87]:


predknr2= RegKNR2.predict(X_test2)


# In[88]:


predknr2


# In[89]:


predknr2=predknr2.astype(int)


# In[90]:


SKN2=RegKNR2.score(X_test2, y_test2)
SKN2=SKN2*100
SKN2


# In[91]:


predknr2= predknr2.reshape(-1,1)
y_test2= y_test2.values.reshape(-1,1)
ConKN2= np.concatenate((y_test2,predknr2),axis=1)


# In[92]:


dfkn2= pd.DataFrame(ConKN2,columns=['Total Arrest(Real)','Prediction by KNR'])


# In[93]:


dfkn2.head(20)


# In[ ]:





# In[94]:


regrf2= RandomForestRegressor(n_estimators= 40, random_state=1)


# In[95]:


regrf2.fit(X_train2, y_train2)


# In[96]:


predrf2= regrf2.predict(X_test2)


# In[97]:


predrf2=predrf2.astype(int)


# In[98]:


Srf2=regrf2.score(X_test2, y_test2)
srf2= Srf2*100
srf2


# In[99]:


predrf2= predrf2.reshape(-1,1)
y_test2= y_test2.reshape(-1,1)
conrf2= np.concatenate((y_test2,predrf2),axis=1)


# In[100]:


dfrf2= pd.DataFrame(conrf2,columns=['Total Arrest (Real)','Prediction RF'])


# In[101]:


dfrf2.head(20)


# In[ ]:





# In[ ]:





# In[102]:


regdt2= DecisionTreeRegressor(random_state=1, max_features=30)
regdt2.fit(X_train2, y_train2)


# In[103]:


preddt2= regdt2.predict(X_test2)


# In[104]:


preddt2=preddt2.astype(int)


# In[105]:


Sdt2=regdt2.score(X_test2, y_test2)
sdt2= Sdt2*100
sdt2


# In[106]:


preddt2= preddt2.reshape(-1,1)
y_test2= y_test2.reshape(-1,1)
condt2= np.concatenate((y_test2,preddt2),axis=1)


# In[107]:


dfdt2= pd.DataFrame(condt2,columns=['Total Arrest(Real)','Prediction by DTR'])


# In[108]:


dfdt2.head(20)


# In[109]:


dfdt22= np.concatenate((X_test2,dfdt2),axis=1)


# In[110]:


dfdt22= pd.DataFrame(dfdt22,columns=['Year', 'T_month', 'Beat', 'Primary Type','Location Description','Count','Total Arrest','Prediction by DTR'])


# In[111]:


dfdt22.head()


# In[ ]:





# In[112]:


scoresofalgorithm = { 'K-Neighbors Regressor': [74.98, 92.75],'Random Forest Regressor': [81.95, 94.03],  'Decision Tree Regressor': [81.63,93.98]}

dfFinal = pd.DataFrame(scoresofalgorithm, index=['Total Crime Prediction','Total Arrest Prediction'])
dfFinal


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




