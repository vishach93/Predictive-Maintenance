#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import streamlit as st 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost
from pickle import dump
from pickle import load


# In[18]:


st.title('Classification of Machine Failure')

st.sidebar.header('User Input Parameters')

def user_input_features():
    TYPE = st.sidebar.selectbox('Type (Quality)- 0: High , 1: Low , 2: Medium',('0','1','2'))
    AIRTEMP = st.sidebar.number_input('Air temperature [K]',min_value=280,max_value=320)
    PROTEMP = st.sidebar.number_input('Process temperature [K]',min_value=280,max_value=320)
    RPM = st.sidebar.number_input('Rotational speed [rpm]',min_value=1000,max_value=3000)
    TRQ = st.sidebar.number_input('Torque [Nm]',min_value=2,max_value=90)
    TW = st.sidebar.number_input('Tool wear [min]',min_value=0,max_value=300)
    data = {'Type':TYPE,
            'Air Temp':AIRTEMP,
            'Process Temp':PROTEMP,
            'Rotational speed':RPM,
            'Torque':TRQ,
            'Tool wear':TW}
    
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# load the model from disk
loaded_model = load(open('Classifier_Model.sav', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
#st.write('Machine Fail' if prediction_proba [0][1] > 0.5 else "Machine won't Fail")
st.write('Machine Will Fail' if prediction == 1 > 0.5 else "Machine Will Not Fail")

#st.subheader('Prediction Probability')
#st.write(prediction_proba, "%.2f")

