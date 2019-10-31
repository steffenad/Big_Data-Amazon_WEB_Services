#!/usr/bin/env python
# coding: utf-8

# ## Big Data - AWS 
# 
# 
# Scale for sentiment towards phone brand:
# 
# 0: very negative
# 
# 1: negative
# 
# 2: somewhat negative
# 
# 3: somewhat positive
# 
# 4: positive
# 
# 5: very positive

# ### Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading Data Sets

# In[753]:


Iphone = pd.read_csv('iphone_smallmatrix.csv')
SamsungGalaxy = pd.read_csv('galaxy_smallmatrix.csv')
LargeMatrix = pd.read_csv('LargeMatrix.csv',sep=';')


# In[754]:


LargeMatrix.drop('id',axis=1,inplace=True)
LargeMatrix.shape


# ### Initial Exploration

# In[9]:


Iphone.head()


# In[10]:


Iphone.info()


# In[15]:


Iphone.describe()


# In[19]:


#any missing values in Data Sets?

print(Iphone.isna().sum().sum())
print(SamsungGalaxy.isna().sum().sum())


# In[687]:


Iphone['iphonesentiment'].hist(bins=20)


# In[688]:


SamsungGalaxy['galaxysentiment'].hist(color='green',bins=20)


# In[755]:


Iphone['iphonesentiment'].replace(1,0,inplace=True)
Iphone['iphonesentiment'].replace(2,3,inplace=True)
Iphone['iphonesentiment'].replace(3,3,inplace=True)
Iphone['iphonesentiment'].replace(4,6,inplace=True)
Iphone['iphonesentiment'].replace(5,6,inplace=True)

SamsungGalaxy['galaxysentiment'].replace(1,0,inplace=True)
SamsungGalaxy['galaxysentiment'].replace(2,3,inplace=True)
SamsungGalaxy['galaxysentiment'].replace(3,3,inplace=True)
SamsungGalaxy['galaxysentiment'].replace(4,6,inplace=True)
SamsungGalaxy['galaxysentiment'].replace(5,6,inplace=True)


# In[643]:


#counting all levels of sentiment towards Iphone in Training Set

Iphone['iphonesentiment'].value_counts(sort=True)


# In[644]:


#counting levels of sentiment for Samsung Galaxy in Training Set

SamsungGalaxy['galaxysentiment'].value_counts(sort=True)


# In[639]:


#average rating for Phone types 

p1 = Iphone['iphonesentiment'].mean()
p2 = SamsungGalaxy['galaxysentiment'].mean()

print('Average Sentiment in Training Data \n\nIphone: {one} \nGalaxy: {two}'.format(one=p1,two=p2))


# ### Correlation Matrix - Iphone

# In[110]:


#detecting collinearity of features

fig, ax = plt.subplots(1, 1, figsize = (16,10))

fig.suptitle('Iphone - Correlation Matrix',fontsize=20)

mask = np.zeros_like(Iphone.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(Iphone.corr(),cmap=cmap,mask=mask,
            square=True, linewidths=.5)


# In[111]:


#detecting collinearity of features

fig, ax = plt.subplots(1, 1, figsize = (16,10))

fig.suptitle('Samsung Galaxy - Correlation Matrix',fontsize=20)

mask = np.zeros_like(SamsungGalaxy.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(SamsungGalaxy.corr(),mask=mask,
            square=True, linewidths=.5)


# ### Feature Selection

# In[412]:


print('Iphone Shape:', Iphone.shape)
print('Samsung Shape:',SamsungGalaxy.shape,'\n')

Iphone_variance = Iphone.var().sort_values()

Samsung_variance = SamsungGalaxy.var().sort_values()

print(Iphone_variance)


# In[735]:


Iphone_novariance = Iphone_variance[Iphone_variance < 1]
Samsung_novariance = Samsung_variance[Samsung_variance <  1]


# In[736]:


Iphone_col_list = Iphone_novariance.index.values.tolist() 
Samsung_col_list = Samsung_novariance.index.values.tolist()


# In[737]:


Iphone.drop(columns = Iphone_col_list, inplace=True)
SamsungGalaxy.drop(columns=Samsung_col_list, inplace=True)


# In[738]:


print('Iphone Shape:', Iphone.shape)
print('Samsung Shape:',SamsungGalaxy.shape)


# In[384]:


#detecting collinearity of features

fig, ax = plt.subplots(1, 1, figsize = (16,10))

fig.suptitle('Iphone - Correlation Matrix',fontsize=20)

mask = np.zeros_like(Iphone.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(Iphone.corr(),cmap=cmap,mask=mask,
            square=True, linewidths=.5)


# In[417]:


Iphone.corr()[Iphone.corr()>0.9]


# ### Modeling Process

# In[626]:


# Classification

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score


# In[756]:


# split Data in training and test sets for IPhone

from sklearn.model_selection import train_test_split

#separate dependent(target) variable and independent variables

target = Iphone.loc[:, Iphone.columns == 'iphonesentiment']

df = Iphone.drop(['iphonesentiment'],axis=1)

#split Data
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=42)


# In[757]:


# split Data in training and test sets for Samsung Galaxy

from sklearn.model_selection import train_test_split

#separate dependent(target) variable and independent variables

target2 = SamsungGalaxy.loc[:, SamsungGalaxy.columns == 'galaxysentiment']

df2 = SamsungGalaxy.drop(['galaxysentiment'],axis=1)

#split Data
X_train2, X_test2, y_train2, y_test2 = train_test_split(df2, target2, test_size=0.25, random_state=42)


# ### SVC (Support Vector Classifier)

# In[445]:


def svc_param_selection(X, y, nfolds):
    Cs = [0.1, 1, 5, 6, 7, 8]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[447]:


import warnings
warnings.filterwarnings('ignore')

svc_param_selection(X_train,y_train, 5)


# In[803]:


# optimal parameter combination for SVM model 
model = SVC(C=5, gamma=0.1, kernel='rbf')


# In[804]:


model.fit(X_train,y_train)


# In[805]:


predictions = model.predict(X_test)


# In[806]:


print(confusion_matrix(y_test,predictions))


# In[807]:


#target_names = ['VERY NEGATIVE', 'NEGATIVE', 'SOMEWHAT NEGATIVE','SOMEWHAT POSITIVE','POSITIVE','VERY POSITIVE']
#target_names=target_names)
print(classification_report(y_test,predictions))


# In[808]:


print('Accuracy: ',accuracy_score(y_test,predictions))
print('Kappa:    ', cohen_kappa_score(predictions,y_test))


# ### KNN

# In[514]:


from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()

def knn_param_selection(X, y, nfolds):
    K = [1,2,3,4,5,6,7,8,9,10]
    
    param_grid = {'n_neighbors': K}
    grid_search = GridSearchCV(knn, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[515]:


knn_param_selection(X_train,y_train, 5)


# In[809]:


#optimal model
model = neighbors.KNeighborsClassifier(n_neighbors = 9)


# In[810]:


model.fit(X_train,y_train)


# In[811]:


predictions = model.predict(X_test)


# In[812]:


print(confusion_matrix(y_test,predictions))


# In[813]:


#target_names = ['VERY NEGATIVE', 'NEGATIVE', 'SOMEWHAT NEGATIVE','SOMEWHAT POSITIVE','POSITIVE','VERY POSITIVE']
#,target_names=target_names
print(classification_report(y_test,predictions))


# In[814]:


print('Accuracy: ',accuracy_score(y_test,predictions))
print('Kappa:    ', cohen_kappa_score(predictions,y_test))


# ### Random Forest Classifier

# In[659]:


from sklearn.ensemble import RandomForestClassifier


# In[476]:


RF = RandomForestClassifier()

def RF_param_selection(X, y, nfolds):
    trees = [700,800,900,950]
    maxfeatures = [2,3,4]
    
    param_grid = {'n_estimators': trees, 'max_features': maxfeatures}
    grid_search = GridSearchCV(RF, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[477]:


RF_param_selection(X_train,y_train, 5)


# In[815]:


#optimal model
model = RandomForestClassifier(n_estimators=700, max_features=4)


# In[816]:


model.fit(X_train,y_train)


# In[817]:


predictions = model.predict(X_test)


# In[818]:


print(confusion_matrix(y_test,predictions))


# In[819]:


target_names = ['VERY NEGATIVE', 'NEGATIVE', 'SOMEWHAT NEGATIVE','SOMEWHAT POSITIVE','POSITIVE','VERY POSITIVE']
#,target_names=target_names)
print(classification_report(y_test,predictions))


# In[820]:


print('Accuracy: ',accuracy_score(y_test,predictions))
print('Kappa:    ', cohen_kappa_score(predictions,y_test))


# ### Neural Network

# In[274]:


from sklearn.neural_network import MLPClassifier


# In[797]:


model = MLPClassifier(hidden_layer_sizes=(50,))


# In[821]:


model.fit(X_train, y_train)


# In[822]:


predictions = model.predict(X_test)


# In[823]:


print(confusion_matrix(y_test,predictions))


# In[824]:


target_names = ['VERY NEGATIVE', 'NEGATIVE', 'SOMEWHAT NEGATIVE','SOMEWHAT POSITIVE','POSITIVE','VERY POSITIVE']
#,target_names=target_names
print(classification_report(y_test,predictions))


# In[825]:


print('Accuracy: ',accuracy_score(y_test,predictions))
print('Kappa:    ', cohen_kappa_score(predictions,y_test))


# In[715]:


#model training for Samsung Galaxy

model2 = MLPClassifier()

model2.fit(X_train2, y_train2)

predictions2 = model2.predict(X_test2)

print(confusion_matrix(y_test2,predictions2))


# In[716]:


target_names = ['VERY NEGATIVE', 'NEGATIVE', 'SOMEWHAT NEGATIVE','SOMEWHAT POSITIVE','POSITIVE','VERY POSITIVE']

print(classification_report(y_test,predictions))
print('Accuracy: ',accuracy_score(y_test2,predictions2))
print('Kappa:    ', cohen_kappa_score(predictions2,y_test2))


# ### Logistic Regression

# In[568]:


from sklearn.linear_model import LogisticRegression


# In[826]:


model = LogisticRegression()


# In[827]:


model.fit(X_train, y_train)


# In[828]:


predictions = model.predict(X_test)


# In[829]:


print(confusion_matrix(y_test,predictions))


# In[830]:


print(classification_report(y_test,predictions))


# In[832]:


print('Accuracy: ',accuracy_score(y_test,predictions))
print('Kappa:    ', cohen_kappa_score(predictions,y_test))


# ### Using Regression instead of Classification for modeling process

# #### Random Forest Regressor

# In[726]:


from sklearn.ensemble import RandomForestRegressor


# In[728]:


model = RandomForestRegressor()

def RF_param_selection(X, y, nfolds):
    trees = [100,200,250,300,400,500]
    
    param_grid = {'n_estimators': trees}
    grid_search = GridSearchCV(RF, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[729]:


RF_param_selection(X_train,y_train, 5)


# In[762]:


#optimal model
model = RandomForestRegressor(n_estimators = 500)


# In[763]:


model.fit(X_train, y_train)


# In[764]:


prediction = model.predict(X_test)


# In[765]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("mean absolute error: ", mean_absolute_error(y_test,prediction))
print("mean squared error: ", mean_squared_error(y_test,prediction))
print("r2_score", r2_score(y_test, prediction))


# ### Making predictions for the Large Data Set

# In[723]:


#optimal model for predictions with accuracy of 84% and Kappa of 60%

model = MLPClassifier()
model.fit(X_train, y_train)

model2 = MLPClassifier()
model2.fit(X_train2, y_train2)


# In[675]:


#predicting sentiment towards Iphone for 3 classes (0=negative,3=neutral,6=positive)

iphonesen_3classes = model.predict(LargeMatrix)


# In[684]:


#predicting sentiment towards Iphone for 3 classes (0=negative,3=neutral,6=positive)

galaxysen_3classes = model2.predict(LargeMatrix)


# In[724]:


#predicting sentiment towards Iphone for 6 classes 

iphonesentiment = model.predict(LargeMatrix)


# In[725]:


#predicting sentiment towards Iphone for 3 classes (0=negative,3=neutral,6=positive)

galaxysentiment = model2.predict(LargeMatrix)


# In[766]:


#including predictions into Data Frame

LargeMatrix['iphonesen_3classes'] = iphonesen_3classes

LargeMatrix['galaxysen_3classes'] = galaxysen_3classes

LargeMatrix['iphonesentiment'] = iphonesentiment

LargeMatrix['galaxysentiment'] = galaxysentiment


# In[767]:


LargeMatrix.shape


# In[769]:


LargeMatrix.head()


# In[770]:


LargeMatrix['iphonesentiment'].hist(bins=20)


# In[775]:


LargeMatrix['galaxysentiment'].hist(color='green',bins=20)


# In[776]:


#counting all levels of sentiment towards Iphone in Training Set

print(LargeMatrix['iphonesentiment'].value_counts(sort=True))
print(LargeMatrix['iphonesen_3classes'].value_counts(sort=True))


# In[777]:


#counting all levels of sentiment towards Iphone in Training Set

print(LargeMatrix['galaxysentiment'].value_counts(sort=True))
print(LargeMatrix['galaxysen_3classes'].value_counts(sort=True))


# In[790]:


#average rating for Phone types 

p1 = LargeMatrix['iphonesentiment'].mean()
p2 = LargeMatrix['galaxysentiment'].mean()

p3 = LargeMatrix['iphonesen_3classes'].mean()
p4 = LargeMatrix['galaxysen_3classes'].mean()

print('Average sentiment towards Iphone and Samsung Galaxy for August 2019 \n\nIphone: {one} \nGalaxy: {two}'.format(one=p1,two=p2))

print('\nClassification with 3 classes: \n')
print('Iphone: {one} \nGalaxy: {two}'.format(one=p3,two=p4))

