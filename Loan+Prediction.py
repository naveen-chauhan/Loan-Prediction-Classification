
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train=pd.read_csv(r'C:\Users\naveen chauhan\Desktop\mldata\loan prediction\train.csv')
train.Loan_Status=train.Loan_Status.map({'Y':1,'N':0})
train.isnull().sum()


# In[3]:


Loan_status=train.Loan_Status
train.drop('Loan_Status',axis=1,inplace=True)
test=pd.read_csv(r'C:\Users\naveen chauhan\Desktop\mldata\loan prediction\test.csv')
Loan_ID=test.Loan_ID
data=train.append(test)
data.head()


# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# In[6]:


data.Dependents.dtypes


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[8]:


data.Gender=data.Gender.map({'Male':1,'Female':0})
data.Gender.value_counts()


# In[9]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[10]:


data.Married=data.Married.map({'Yes':1,'No':0})


# In[11]:


data.Married.value_counts()


# In[12]:


data.Dependents=data.Dependents.map({'0':0,'1':1,'2':2,'3+':3})


# In[13]:


data.Dependents.value_counts()


# In[14]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[15]:


data.Education=data.Education.map({'Graduate':1,'Not Graduate':0})


# In[16]:


data.Education.value_counts()


# In[17]:


data.Self_Employed=data.Self_Employed.map({'Yes':1,'No':0})


# In[18]:


data.Self_Employed.value_counts()


# In[19]:


data.Property_Area.value_counts()


# In[20]:


data.Property_Area=data.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})


# In[21]:


data.Property_Area.value_counts()


# In[22]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[23]:


data.head()


# In[24]:


data.Credit_History.size


# In[25]:


data.Credit_History.fillna(np.random.randint(0,2),inplace=True)


# In[26]:


data.isnull().sum()


# In[27]:


data.Married.fillna(np.random.randint(0,2),inplace=True)


# In[28]:


data.isnull().sum()


# In[29]:


data.LoanAmount.fillna(data.LoanAmount.median(),inplace=True)


# In[30]:


data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(),inplace=True)


# In[31]:


data.isnull().sum()


# In[32]:


data.Gender.value_counts()


# In[33]:


from random import randint 
data.Gender.fillna(np.random.randint(0,2),inplace=True)


# In[34]:


data.Gender.value_counts()


# In[35]:


data.Dependents.fillna(data.Dependents.median(),inplace=True)


# In[36]:


data.isnull().sum()


# In[37]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[38]:


data.Self_Employed.fillna(np.random.randint(0,2),inplace=True)


# In[39]:


data.isnull().sum()


# In[40]:


data.head()


# In[41]:


data.drop('Loan_ID',inplace=True,axis=1)


# In[42]:


data.isnull().sum()


# In[43]:


train_X=data.iloc[:614,]
train_y=Loan_status
X_test=data.iloc[614:,]
seed=7


# In[44]:


from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(train_X,train_y,random_state=seed)


# In[45]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[46]:


models=[]
models.append(("logreg",LogisticRegression()))
models.append(("tree",DecisionTreeClassifier()))
models.append(("lda",LinearDiscriminantAnalysis()))
models.append(("svc",SVC()))
models.append(("knn",KNeighborsClassifier()))
models.append(("nb",GaussianNB()))


# In[47]:


seed=7
scoring='accuracy'


# In[48]:


from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
result=[]
names=[]


# In[49]:


for name,model in models:
    #print(model)
    kfold=KFold(n_splits=10,random_state=seed)
    cv_result=cross_val_score(model,train_X,train_y,cv=kfold,scoring=scoring)
    result.append(cv_result)
    names.append(name)
    print("%s %f %f" % (name,cv_result.mean(),cv_result.std()))


# In[50]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
svc=LogisticRegression()
svc.fit(train_X,train_y)
pred=svc.predict(test_X)
print(accuracy_score(test_y,pred))
print(confusion_matrix(test_y,pred))
print(classification_report(test_y,pred))


# In[51]:


df_output=pd.DataFrame()


# In[52]:


outp=svc.predict(X_test).astype(int)
outp


# In[53]:


df_output['Loan_ID']=Loan_ID
df_output['Loan_Status']=outp


# In[54]:


df_output.head()


# In[56]:


df_output[['Loan_ID','Loan_Status']].to_csv(r'C:\Users\naveen chauhan\Desktop\mldata\loan prediction\output.csv',index=False)

