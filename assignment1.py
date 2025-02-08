# Importing Modules
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing
import pandas as pd

#Preparing Training Data
trainData = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
trainData = trainData['trips']
model = SimpleExpSmoothing(trainData)
modelFit = model.fit(smoothing_level=0.5,optimized=False)
# modelFit = model.fit(optimized=True)
pred = modelFit.forecast(744)