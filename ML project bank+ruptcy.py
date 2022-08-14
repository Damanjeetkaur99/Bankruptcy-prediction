#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ## Loading the required dataset

# In[2]:


bank=pd.read_csv("bankrup.csv",sep=';')
df=pd.DataFrame(bank)
df


# ### This dataset is about prevention of bankruptcy it includes cols as:
# #### 1. Industrial_Risk:- It is on of the risk that drags down company's performance.
# #### 2.Management_Risk:- It is the process of identifying, analysing, and accepting of uncertainty in industry.
# #### 3.Financial_Flexibility:- It is the ability of a firm to access and restructure its financing at a low cost.
# #### 4.Credibility:- It is the commitment to follow well-articulated and transparent rules.
# #### 5. Competitiveness:- It refers to the policies and laws that affect the extent to which banks compete.
# #### 6. Operating_Risk:- It refers to the risk of loss resulting from inadequate or failed internal processes, people, and systems or from external affairs.
# #### 7. Class:- This column includes the two classes: a) Bankruptcy ; b) Non-Bankruptcy.

# ## To Prevent Bankruptcy industrial_risk, management risk and operating risk should be low, Whereas financial flexibility, credibility and competitiveness should be high

# In[3]:


df.info()


# #### checking for null values

# In[4]:


df.isnull().sum()


# #### checking for datatypes

# In[5]:


df.dtypes


# #### checking for no. of rows and cols

# In[6]:


df.shape


# In[7]:


df=df.rename(columns={' class':'class',' management_risk':'management_risk',' financial_flexibility':'financial_flexibility',' credibility':'credibility',' competitiveness':'competitiveness',' operating_risk':'operating_risk'})


# In[8]:


data=df.copy()


# In[9]:


data


# ### descriptive info for the data

# In[10]:


data.describe()


# ### Transforming class column as 0 for Bankruptcy and 1 for Non-Bankruptcy

# In[11]:


labelencoder=LabelEncoder()
data.iloc[:,6]=labelencoder.fit_transform(data.iloc[:,6])


# In[12]:


data


# In[13]:


data['class'].value_counts()


# In[14]:


org_data=data.copy()


# In[15]:


label='Non-Bankruptcy','Bankruptcy'
color='green','red'
data['class'].value_counts().plot(kind="pie",autopct='%1.1f%%',labels=label,shadow=True,colors=color)
plt.axis('equal')
plt.show()


# ##### finding correlation between features

# In[15]:


data_corr=data.corr()
data_corr


# In[18]:


sns.set_style(style='darkgrid')
sns.pairplot(data)


# In[16]:


sns.heatmap(data_corr)


# ##### So, from above corr method we  found correlation between risk and class features 

# #### visualizing data using density plot

# In[78]:


sns.countplot(data['class'])


# In[105]:


sns.kdeplot(data=data,x='industrial_risk',hue='class',multiple="fill")


# In[92]:


sns.kdeplot(data=data,x='management_risk',hue='class',multiple="fill")


# In[93]:


sns.kdeplot(data=data,x='operating_risk',hue='class',multiple="fill")


# In[16]:


sns.kdeplot(data=data,x='credibility',hue='class',multiple="fill")


# In[95]:


sns.kdeplot(data=data,x='financial_flexibility',hue='class',multiple="fill")


# In[96]:


sns.kdeplot(data=data,x='competitiveness',hue='class',multiple="fill")


# In[ ]:





# ### Outlier Detection

# #### Using isolation forest to detect outlier.
# #### we can also use cook's distance to detect the same

# In[17]:


from sklearn.ensemble import IsolationForest


# In[18]:


data1=data.iloc[:,0:6]
data1.head()


# In[19]:


data_col=data1.copy()
cols=data_col.columns


# In[20]:


clf=IsolationForest(random_state=10,contamination=.01)
clf.fit(data1)


# In[21]:


data_outlier=clf.predict(data1)
data_outlier


# ##### Here 1 for inliners and -1 is for outliers

# In[22]:


data1['scores']=clf.decision_function(data1)
data1['anomaly']=clf.predict(data1.iloc[:,0:6])
data1


# ### Outliers in data

# In[23]:


data1[data1['anomaly']==-1]


# In[24]:


#dropping outliers


# In[25]:


#data=data.drop(data.index[[27,7,192]],axis=0)


# In[26]:


#reseting index


# In[27]:


#data.reset_index(drop=True,inplace=True)
#data.head()


# In[28]:


data_f=data.iloc[:,0:7]
data_f


# ## Performing Clustering to find patterns in the data

# #### Hierarchical Clustering

# In[29]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# ##### Creating Dendrogram

# In[28]:


bank_dend=sch.dendrogram(sch.linkage(data,method='single'))


# ##### observing above dendrogram we should take n_clusters=3

# ##### Forming clusters

# In[29]:


bank_heir_cc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='single')
y_cc=bank_heir_cc.fit_predict(data)


# In[30]:


clusters_heir=pd.DataFrame(y_cc,columns=['Clusters'])
clusters_heir.head()


# In[31]:


data['h_clusterid']=clusters_heir
data


# #### sorting values on the basis of clusters created

# In[32]:


hc=data.sort_values('h_clusterid')
hc


# #### K-Means Clustering

# In[33]:


from sklearn.cluster import KMeans


# ##### using WCSS (elbow method) method to identify n_clusters 

# In[34]:


wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=0)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)#inertia means twss


# In[30]:


plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS')
plt.show()


# In[36]:


wcss


# ### From the 2nd cluster i.e. 160.64 the drop is way more constant so, n_clusters=2 would be better one

# In[37]:


kmeans_clusters=KMeans(2,random_state=98)
kmeans_clusters.fit(data)


# In[38]:


kmeans_clusters.labels_


# In[39]:


data['k_clusterid']=kmeans_clusters.labels_
data


# #### standardizing values

# In[40]:


kmeans_clusters.cluster_centers_


# #### Means of clusters

# In[41]:


data.groupby('k_clusterid').agg(['mean']).reset_index()


# In[42]:


data2=data.iloc[:,0:7]
data2.head()


# #### DBSCAN

# In[43]:


from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# #### we will use knn to determine optimum no. of epsilon(eps)

# In[44]:


neighbors=NearestNeighbors(n_neighbors=14)
neighbors_fit=neighbors.fit(data2)
distances,indices=neighbors_fit.kneighbors(data2)


# #### sorting values by ascending value and plot

# In[45]:


distances=np.sort(distances,axis=0)
distances=distances[:,1]
plt.plot(distances)


# #### taking 0.8 as eps

# #### DBSCAN algo

# In[46]:


dbscan=DBSCAN(eps=.8,min_samples=14)
y_pred=dbscan.fit(data2)
y_pred


# #### 0 and 1 are clusters and -1 are noise

# In[47]:


labels=dbscan.labels_
labels


# In[48]:


cls=pd.DataFrame(labels,columns=['DS_clusterid'])
cls


# In[49]:


db_df=pd.concat([data2,cls],axis=1)
db_df


# In[50]:


data3=pd.concat([data,cls],axis=1)
data3


# ### So, by performing DBSCAN 2 Clusters are identified

# In[ ]:





# # Feature Extraction

# ### Using Logistic Regression

# In[31]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[32]:


org_data


# In[33]:


data1.columns


# In[34]:


array=org_data.values


# #### spliting into input and output 

# In[35]:


x=array[:,0:6]
y=array[:,-1]


# #### fitting into RFE model 

# In[36]:


model_rfe=LogisticRegression(max_iter=400)
rfe=RFE(model_rfe,4)
fit=rfe.fit(x,y)


# #### no. of features

# In[37]:


fit.n_features_


# #### extracting top 4 most imp features

# In[38]:


fit.support_


# #### extracting 4 top most imp features by ranking them

# In[39]:


fit.ranking_


# ### So, from the above we concluded that the top most 4 imp features are : management_risk, financial_flexibility, credibility, compitetiveness

# ## Using Decision Tree

# In[40]:


from sklearn.tree import  DecisionTreeClassifier


# In[41]:


model_dt=DecisionTreeClassifier()
fit_model=model_dt.fit(x,y)
fit_model


# In[42]:


fea=fit_model.feature_importances_
fea


# In[43]:


cols


# In[44]:


fea_df=pd.DataFrame(fea,columns=['feature_imp'])
col_df=pd.DataFrame(cols,columns=['columns'])


# In[45]:


fea_imp_df=pd.concat([col_df,fea_df],axis=1)
fea_imp_df


# ## According to this Financial_Flexibility and competetiveness are the most imp features

# In[ ]:





# # Model Building 

# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


org_data


# In[48]:


X=org_data.iloc[:,0:6]
Y=org_data['class']


# #### spliting data into train and test

# In[49]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.2,random_state=12)


# In[50]:


from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# ### Using EnsembleTechniques to reduce complexity of data

# In[51]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()


# In[52]:


params={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_reg=GridSearchCV(lasso,params,scoring='neg_mean_squared_error',cv=5)
lasso_reg.fit(x_train,y_train)
print(lasso_reg.best_params_)
print(lasso_reg.best_score_)


# In[53]:


params={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_reg=GridSearchCV(lasso,params,scoring='neg_mean_squared_error',cv=5)
lasso_reg.fit(x_test,y_test)
print(lasso_reg.best_params_)
print(lasso_reg.best_score_)


# #### Lasso

# In[54]:


lasso=Lasso(alpha=0.01)
lasso.fit(x_train,y_train)
y_pred_lasso=lasso.predict(x_train)


# In[55]:


mean_squared_error = np.mean((y_pred_lasso - y_train)**2) 
print("Mean squared error on test set", mean_squared_error) 


# In[56]:


lasso_coeff = pd.DataFrame() 
lasso_coeff["Columns"] = x_train.columns 
lasso_coeff['Coefficient Estimate'] = pd.Series(lasso.coef_) 
print(lasso_coeff) 


# In[57]:


lasso=Lasso(alpha=0.001)
lasso.fit(x_test,y_test)
y_pred_lasso_test=lasso.predict(x_test)


# In[58]:


mean_squared_error = np.mean((y_pred_lasso_test - y_test)**2) 
print("Mean squared error on test set", mean_squared_error) 


# In[59]:


lasso_coeff = pd.DataFrame() 
lasso_coeff["Columns"] = x_test.columns 
lasso_coeff['Coefficient Estimate'] = pd.Series(lasso.coef_) 
print(lasso_coeff) 


# #### Ridge

# In[60]:


ridgeR = Ridge(alpha = .01) 
ridgeR.fit(x_train, y_train) 
y_pred_ridge = ridgeR.predict(x_test) 


# In[61]:


mean_squared_error_ridge = np.mean((y_pred_ridge - y_test)**2) 
print(mean_squared_error_ridge) 


# In[62]:


ridge_coefficient = pd.DataFrame() 
ridge_coefficient["Columns"]= x_train.columns 
ridge_coefficient['Coefficient Estimate'] = pd.Series(ridgeR.coef_) 
print(ridge_coefficient) 


# #### Elastic Net

# In[63]:


e_net = ElasticNet(alpha = 0.001,l1_ratio=.5) 
e_net.fit(x_train, y_train)


# In[64]:


y_pred_elastic = e_net.predict(x_test) 
mean_squared_error = np.mean((y_pred_elastic - y_test)**2) 
print("Mean Squared Error on test set", mean_squared_error) 


# In[65]:


e_net_coeff = pd.DataFrame() 
e_net_coeff["Columns"] = x_train.columns 
e_net_coeff['Coefficient Estimate'] = pd.Series(e_net.coef_) 
e_net_coeff 


# #### XGBoost

# In[66]:


from xgboost import XGBClassifier


# In[67]:


model = XGBClassifier()
model.fit(x_train, y_train)


# In[68]:


y_pred = model.predict(x_test)


# In[69]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# #### AdaBoost

# In[70]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

kfold=KFold(n_splits=10)
num_trees=100
model_a=AdaBoostClassifier(n_estimators=num_trees,learning_rate=0.8,random_state=8)
results=cross_val_score(model_a,x_train,y_train,cv=kfold)
BOOST=results.mean()
BOOST


# ### Logistic Regression

# In[71]:


from sklearn.linear_model import LogisticRegression


# ##### training data

# In[72]:


lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[73]:


y_pred_train_lr=lr.predict(x_train)
acc_train_lr=accuracy_score(y_train,y_pred_train_lr)
acc_train_lr


# ##### test data

# In[74]:


y_pred_test_lr=lr.predict(x_test)
acc_test_lr=accuracy_score(y_test,y_pred_test_lr)
acc_test_lr


# #### cross val score

# In[75]:


score=cross_val_score(lr,x_train,y_train,cv=5)
np.mean(score)*100


# In[76]:


new={'industrial_risk':0.0,
    'management_risk':1.0,
    'financial_flexibility':0.0,
    'credibility':1.0,
    'competitiveness':1.0,
    'operating_risk':0.0}


# In[77]:


df4=pd.DataFrame(new,index=[0])


# In[78]:


lr.predict(df4)


# In[79]:


import pickle


# In[80]:


pickle_out=open('C:/vinnu/p138.pkl','wb')


# In[81]:


pickle.dump(lr,pickle_out)
pickle_out.close()


# ### Decision Tree

# In[58]:


from sklearn.tree import DecisionTreeClassifier


# ##### training data

# In[59]:


dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)


# ##### train data

# In[60]:


y_pred_train_dt=dt.predict(x_train)
acc_train_dt=accuracy_score(y_train,y_pred_train_dt)
acc_train_dt


# #### test data

# In[210]:


y_pred_test_dt=dt.predict(x_test)
acc_test_dt=accuracy_score(y_test,y_pred_test_dt)
acc_test_dt


# #### cross val score

# In[83]:


score=cross_val_score(dt,x_train,y_train,cv=5)
np.mean(score)*100


# ### Kneighbors

# In[63]:


from sklearn.neighbors import KNeighborsClassifier


# ##### training data

# In[64]:


neighbors=KNeighborsClassifier(n_neighbors=14)
neighbors.fit(x_train,y_train)


# ##### training data

# In[65]:


y_pred_train_neighbors=neighbors.predict(x_train)
acc_train_neighbors=accuracy_score(y_train,y_pred_train_neighbors)
acc_train_neighbors


# ##### test data

# In[66]:


y_pred_test_neighbors=neighbors.predict(x_test)
acc_test_neighbors=accuracy_score(y_test,y_pred_test_neighbors)
acc_test_neighbors


# #### cross val score

# In[84]:


score=cross_val_score(neighbors,x_train,y_train,cv=5)
np.mean(score)*100


# ### Neural Networks

# In[68]:


import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout


# In[69]:


model_nn=Sequential()
model_nn.add(Dense(8,input_dim=6,activation='relu'))
model_nn.add(Dropout(.5))
model_nn.add(Dense(6,activation='relu'))
model_nn.add(Dense(1,activation='sigmoid'))


# In[70]:


model_nn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# ##### training data

# #### using epochs=100 and batch_size=30

# In[71]:


model_nn.fit(x_train,y_train,validation_split=0.33,epochs=100,batch_size=30)


# In[72]:


scores=model_nn.evaluate(x_train,y_train)
print("%s: %.2f%%" % (model_nn.metrics_names[1], scores[1]*100))


# ##### test data

# In[73]:


model_nn.fit(x_test,y_test,validation_split=0.33,epochs=100,batch_size=30)


# In[74]:


scores=model_nn.evaluate(x_test,y_test)
print("%s: %.2f%%" % (model_nn.metrics_names[1], scores[1]*100))


# In[75]:


from sklearn.model_selection import cross_val_score
from keras.layers import InputLayer


# In[76]:


def create_model():
    model1=Sequential()
    model1.add(Dense(10,input_dim=6,activation='tanh'))
    model1.add(Dense(7,activation='tanh'))
    model1.add(Dense(1))
    model1.compile(loss='mean_squared_error',optimizer='adam')
    return model1


# In[77]:


from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor


# #### using epochs=50 and batch_size=10

# In[78]:


model2=KerasRegressor(build_fn=create_model,epochs=50,batch_size=10,verbose=False)
kfold=KFold(n_splits=10)
results=(cross_val_score(model2,x_train,y_train,cv=kfold))
print("Results:%.2f(%.2f) MSE" %(results.mean(),results.std()))


# In[79]:


model2.fit(x_train,y_train)


# In[80]:


pred_train=model2.predict(x_train)


# In[81]:


model3=KerasRegressor(build_fn=create_model,epochs=50,batch_size=10,verbose=False)
kfold=KFold(n_splits=10)
results=(cross_val_score(model3,x_test,y_test,cv=kfold))
print("Results:%.2f(%.2f) MSE" %(results.mean(),results.std()))


# In[85]:


model3.fit(x_test,y_test)


# In[86]:


pred_test=model3.predict(x_test)


# ### Random Forest

# In[87]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,precision_score, accuracy_score


# In[88]:


rfc=RandomForestClassifier(max_depth=6)
rfc.fit(x_train,y_train)


# ##### training data

# In[89]:


y_pred_train_rfc=rfc.predict(x_train)
acc_train_rfc=accuracy_score(y_train,y_pred_train_rfc)
acc_train_rfc


# In[90]:


confusion_matrix(y_train,y_pred_train_rfc)


# In[91]:


precision_score(y_train,y_pred_train_rfc)


# ##### test data

# In[92]:


y_pred_test_rfc=rfc.predict(x_test)
acc_test_rfc=accuracy_score(y_test,y_pred_test_rfc)
acc_test_rfc


# In[93]:


confusion_matrix(y_test,y_pred_test_rfc)


# In[94]:


precision_score(y_test,y_pred_test_rfc)


# #### cross_val_score

# In[96]:


score=cross_val_score(rfc,x_train,y_train,cv=5)
np.mean(score)*100


# ### SVM

# In[95]:


from sklearn.svm import SVC


# #### kernel=Linear

# ##### train data

# In[102]:


svm_lin=SVC(kernel='linear',C=.5)
svm_lin.fit(x_train,y_train)
svm_lin_train_pred=svm_lin.predict(x_train)
svm_lin_train_acc=accuracy_score(y_train,svm_lin_train_pred)
svm_lin_train_acc


# ##### test data

# In[103]:


svm_lin_test_pred=svm_lin.predict(x_test)
svm_lin_test_acc=accuracy_score(y_test,svm_lin_test_pred)
svm_lin_test_acc


# #### cross_val_score

# In[108]:


score=cross_val_score(svm_lin,x_train,y_train,cv=5)
np.mean(score)*100


# #### kernel=Poly

# ##### train data

# In[104]:


svm_poly=SVC(kernel='poly',C=.5)
svm_poly.fit(x_train,y_train)
svm_poly_train_pred=svm_poly.predict(x_train)
svm_poly_train_acc=accuracy_score(y_train,svm_poly_train_pred)
svm_poly_train_acc


# ##### test data

# In[106]:


svm_poly_test_pred=svm_poly.predict(x_test)
svm_poly_test_acc=accuracy_score(y_test,svm_poly_test_pred)
svm_poly_test_acc


# #### cross_val_score

# In[107]:


score=cross_val_score(svm_poly,x_train,y_train,cv=5)
np.mean(score)*100

