#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


# In[2]:


os.chdir("F:/analytics_basics/bike_prediction")


# In[3]:


data=pd.read_csv("day.csv")


# In[4]:


data.head()


# In[5]:


data.dtypes


# In[6]:


data.nunique()


# In[7]:


cat_cnames = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
num_cnames = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']


# In[8]:


data[num_cnames].describe()


# In[9]:


data.isnull().sum()


# In[10]:


fig0, ax0 = plt.subplots(nrows = 1, ncols = 4, figsize= (15,5))
var = ['temp', 'atemp', 'hum', 'windspeed']
i=0
for j in var:
    ax1 = sns.distplot((data[j]),ax = ax0[i], rug=True, color="b")
    ax1.set_title('Distribution plot - ' +str(j))
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel(str(j))
    i+=1


# In[11]:


for i in cat_cnames:
    print(i)
    print(data[i].value_counts())
    print("   ")


# In[12]:


data.groupby('yr')['cnt'].sum().plot.bar(x='yr',y='cnt')


# In[13]:


var2 = ['season', 'mnth', 'weekday', 'weathersit']
j=0
for i in var2:
    df = data.groupby(i)['cnt'].sum()
    df = df.reset_index()
    df.plot.bar(x=i,y='cnt').set_title('Sum plot - ' +str(i))
    gp = data.groupby(by = ['yr', i]).sum().reset_index()
    (sns.catplot(x= i, y = 'cnt', data = gp, col = 'yr', kind = 'bar'))
    j+=1


# In[14]:


from pandas.tools.plotting import scatter_matrix
scatter_matrix(data[num_cnames], alpha=0.2, figsize=(15, 15), diagonal='kde')


# In[15]:


corr = data[num_cnames].corr()
corr
corr.style.background_gradient()


# In[16]:


# making every combination from cat_cnames
factors_paired = [(i,j) for i in cat_cnames for j in cat_cnames] 
# doing chi-square test for every combination
p_values = []
from scipy.stats import chi2_contingency
for factor in factors_paired:
    if factor[0] != factor[1]:
        chi2, p, dof, ex = chi2_contingency(pd.crosstab(data[factor[0]], 
                                                    data[factor[1]]))
        p_values.append(p.round(3))
    else:
        p_values.append('-')
p_values = np.array(p_values).reshape((7,7))
p_values = pd.DataFrame(p_values, index=cat_cnames, columns=cat_cnames)
print(p_values)


# In[17]:



#ANOVA Analysis
import statsmodels.api as sm
from statsmodels.formula.api import ols


cw_lm=ols('cnt ~ C(yr)+C(holiday)+C(workingday)+ C(mnth)+C(weekday)+ C(weathersit)+C(season)', data=data).fit() 
print(sm.stats.anova_lm(cw_lm, typ=2))


# In[18]:


# Making dummies
season_dm = pd.get_dummies(data['season'], drop_first=True, prefix='season')
data = pd.concat([data, season_dm],axis=1)
data = data.drop(columns = ['season'])
weather_dm = pd.get_dummies(data['weathersit'], prefix= 'weather',drop_first=True)
data = pd.concat([data, weather_dm],axis=1)
data = data.drop(columns= ['weathersit'])


# In[19]:


# creating another dataset with removed outliers
# data_wo (data without outliers)- for checking and comparing the performance with the 
data_wo = data.copy()
# dropping outliers from boxplot method
for i in ['windspeed', 'hum']:
    q75, q25 = np.percentile(data_wo.loc[:,i], [75 ,25])
    iqr = q75 - q25
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    data_wo = data_wo.drop(data_wo[data_wo.loc[:,i] < min].index)
    data_wo = data_wo.drop(data_wo[data_wo.loc[:,i] > max].index)


# In[20]:


# dropping unwanted columns as decided in feature selection.
data.drop(columns=['instant', 'dteday', 'holiday', 'atemp', 'casual', 'registered'], inplace=True)
data_wo.drop(columns=['instant', 'dteday', 'holiday', 'atemp', 'casual', 'registered'], inplace=True)

data.head()


# In[77]:


####******** Building Machine learning models *******#####

# fuction to check the performance of the regression model using kfold cross validation on explained variance
# also checking the score with the training and test dataset
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
def predictions(regressor_model, X_train, y_train, X_test, y_test):
    regressor_model.fit(X_train, y_train)
    y_pred1 = regressor_model.predict(X_train)
    y_pred2 = regressor_model.predict(X_test)
    # here we are taking the k fold parameter as 10. It will divide the whole dataset into 10 equal parts and check performance taking each part one time as test data and other parts as training data
    performance = cross_val_score(estimator=regressor_model, X = X_train, y = y_train, cv = 10, 
                                       scoring='neg_mean_absolute_error')

    k_fold_performance = -(performance.mean())
    print("K-fold MAE")
    print(k_fold_performance)
    print()
    print("training data MAE")
    print(mean_absolute_error(y_pred1, y_train)) 
    print()
    print("test data MAE")
    print(mean_absolute_error(y_pred2, y_test))


# In[78]:


# splitting dataset in train and test for whole dataset
X = data.drop(columns=['cnt'])
y = data['cnt']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[79]:


# splitting dataset in train and test for dataset without outlier
X = data_wo.drop(columns=['cnt'])
y = data_wo['cnt']
from sklearn.model_selection import train_test_split
X_train_wo, X_test_wo, y_train_wo, y_test_wo = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[80]:


#   Linear Regression   #

# building model for dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
print("WITH INCLUSION OF OUTLIERS")

predictions(regressor, X_train, y_train, X_test, y_test)
print()

# building model for dataset without  outliers
regressor = LinearRegression()
print()
print("WITHOUT OUTLIERS")
predictions(regressor, X_train_wo, y_train_wo, X_test_wo, y_test_wo)


# In[83]:


#         KNN           #

# building model for dataset bike_data
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=5)
print("WITH INCLUSION OF OUTLIERS")
predictions(regressor, X_train, y_train, X_test, y_test)
print()

# building model for dataset  without outliers
regressor = KNeighborsRegressor(n_neighbors=5)
print("WITHOUT OUTLIERS")
predictions(regressor, X_train_wo, y_train_wo, X_test_wo, y_test_wo)


# In[84]:


#        SVM            #

# building model for dataset bike_data
from sklearn.svm import SVR
regressor = SVR()
print("WITH INCLUSION OF OUTLIERS")
predictions(regressor, X_train, y_train, X_test, y_test)
print()

# building model for dataset bike_data_wo i.e. without outliers
regressor = SVR()
print("WITHOUT OUTLIERS")
predictions(regressor, X_train_wo, y_train_wo, X_test_wo, y_test_wo)


# In[92]:


# Decision Tree Regression  #

# building model for dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=1)
print("WITH INCLUSION OF OUTLIERS")
predictions(regressor, X_train, y_train, X_test, y_test)
print()

# building model for dataset without outliers
print("WITHOUT OUTLIERS")
predictions(regressor, X_train_wo, y_train_wo, X_test_wo, y_test_wo)


# In[88]:


#  Random Forest   #

# building model for dataset bike_data
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(random_state=1)
print("WITH INCLUSION OF OUTLIERS")
predictions(regressor, X_train, y_train, X_test, y_test)
print()
# building model for dataset bike_data_wo i.e. without outliers
regressor = RandomForestRegressor(random_state=1)
print("WITHOUT OUTLIERS")
predictions(regressor, X_train_wo, y_train_wo, X_test_wo, y_test_wo)


# In[90]:


#     XGBRegressor      #

# building model for dataset bike_data
from xgboost import XGBRegressor
regressor = XGBRegressor(random_state=1)
print("WITH INCLUSION OF OUTLIERS")
predictions(regressor, X_train, y_train, X_test, y_test)
print()

# building model for dataset bike_data_wo i.e. without outliers
regressor = XGBRegressor(random_state=1)
print('WITHOUT OUTLIERS')
predictions(regressor, X_train_wo, y_train_wo, X_test_wo, y_test_wo)


# In[91]:


#        Hyperparameter tuning             #

# tuning Random Forest for dataset #

from sklearn.model_selection import GridSearchCV
# Random Forest hyperparameter tuning
regressor = RandomForestRegressor(random_state=1)
params = [{'n_estimators' : [500, 600, 800],'max_features':['auto', 'sqrt', 'log2'],
           'min_samples_split':[2,4,6],'max_depth':[12, 14, 16],'min_samples_leaf':[2,3,5],
           'random_state' :[1]}]
grid_search = GridSearchCV(estimator=regressor, param_grid=params,cv = 5,
                           scoring = 'explained_variance', n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
print(grid_search.best_params_)


# In[97]:


# building Random Forest on tuned parameter
regressor = RandomForestRegressor(random_state=1, max_depth=16, n_estimators=600,
                                  max_features='auto', min_samples_leaf=2,min_samples_split=2)
predictions(regressor, X_train, y_train, X_test, y_test)


# In[94]:


# tuning XGBRegressor for dataset   #

regressor = XGBRegressor(random_state=1)
params = [{'n_estimators' : [250, 300,350, 400,450], 'max_depth':[2, 3, 5], 
           'learning_rate':[0.01, 0.045, 0.05, 0.055, 0.1, 0.3],'gamma':[0, 0.001, 0.01, 0.03],
           'subsample':[1, 0.7, 0.8, 0.9],'random_state' :[1]}]
grid_search = GridSearchCV(estimator=regressor, param_grid=params,cv = 5,
                           scoring = 'explained_variance', n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
print(grid_search.best_params_)


# In[100]:


# Building XGBRegressor on tuned parameter
regressor = XGBRegressor(random_state=1, learning_rate=0.055, max_depth=3, n_estimators=250, 
                         gamma = 0, subsample=0.7)
predictions(regressor, X_train, y_train, X_test, y_test)


# In[ ]:




