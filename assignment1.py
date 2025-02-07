# Importing Modules
import pandas as pd, numpy as np
from prophet import Prophet
import plotly.express as px

#Preparing Training Data
trainData = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
trainData = trainData[['Timestamp','trips']]
trainData.columns = ['ds', 'y']

#Loading and fitting model
model = Prophet(changepoint_prior_scale = 0.5)
modelFit = model.fit(trainData)

#Preparing Test Data
testData = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")
testData = testData[['Timestamp']]
testData.columns = ['ds']

#Getting Forecasting Data
forecast = model.predict(testData)

pred = forecast['yhat']