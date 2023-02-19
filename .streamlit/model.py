#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
import warnings 
warnings.filterwarnings('ignore')
import xgboost
from pickle import dump
from pickle import load


# In[21]:


df = pd.read_csv('Maintenance.csv')


label= LabelEncoder()
df['Type']= label.fit_transform(df['Type'])

outlier_model=IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.2))
outlier_model.fit(df[['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]',
           'Torque [Nm]','Tool wear [min]']])

df['scores'] = outlier_model.decision_function(df[['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]',
           'Torque [Nm]','Tool wear [min]']])
df['anomaly_score'] = outlier_model.predict(df[['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]',
           'Torque [Nm]','Tool wear [min]']])

droped_df= df.drop(df[(df['Machine failure']==0) & (df['anomaly_score']==-1)].index)

renamed_df = droped_df.rename(columns={"Air temperature [K]": "Air_temperature", "Process temperature [K]": "Process_temperature", "Rotational speed [rpm]": "Rotational_speed","Torque [Nm]":"Torque", "Tool wear [min]":"Tool_wear"}, errors="raise")


X=renamed_df.iloc[:,[2,3,4,5,6,7,9,10,11,12,13]]
y=renamed_df.iloc[:,8]

smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)

x_smote, y_smote = smote.fit_resample(X, y)

processed_df = x_smote
processed_df['Machine failure'] = y_smote

X=processed_df.iloc[:,:6]
y=processed_df.iloc[:,11]

classifier_X=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7, gamma=0.4, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.25, max_delta_step=0, max_depth=15,
              min_child_weight=3, monotone_constraints='()',
              n_estimators=100, n_jobs=12, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)


classifier_X.fit(X,y)


# In[24]:


dump(classifier_X,open('Classifier_Model.sav', 'wb'))

loaded_model=load(open('Classifier_Model.sav' ,'rb'))
result = loaded_model.score(X,y)
print(result)

