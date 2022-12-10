from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Reading the data 
raw_data= pd.read_csv('adult.csv')

# Creating labels  
X= raw_data.iloc[:,:-1]
y=raw_data.iloc[:,-1:]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Creating Pipelines 
numeric_pipeline = Pipeline(steps=[('impute',SimpleImputer(strategy='mean')),('scale',MinMaxScaler())])
categorical_pipeline = Pipeline(steps=[('impute',SimpleImputer(strategy='most_frequent')),('onehotencode',OneHotEncoder(handle_unknown='ignore',sparse=False))])


# Getting the numerical and categorical features 
numerical_features= X_train.select_dtypes(include='number').columns.tolist()
categorical_features= X_train.select_dtypes(exclude='number').columns.tolist()


# Creating Machine Learning Pipeline 
from sklearn.compose import ColumnTransformer
full_processor= ColumnTransformer(transformers=[('number',numeric_pipeline,numerical_features),('object',categorical_pipeline,categorical_features)])


# Fitting the data
full_processor.fit_transform(X_train)

# Creating the final ML Pipeline 
from sklearn.ensemble import RandomForestClassifier
rcl_pipeline= Pipeline(steps=[('preprocess',full_processor),('randomforest', RandomForestClassifier(n_estimators=10))])

# Fitting the  training data 
rcl_pipeline.fit(X_train,y_train)

# Scoring the model 
pred_vals=rcl_pipeline.predict(X_test)

# Evaluating the model 
model_score= rcl_pipeline.score(X_train,y_train)
print("model_score", model_score)

from sklearn.metrics import accuracy_score

score= accuracy_score(y_test,pred_vals)
print("accuracy_score",score)