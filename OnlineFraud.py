import pandas as pd
import numpy as np
data = pd.read_csv("credit card.csv")
print(data.head())
print(data.isnull().sum())#If the've any null values...It'll print them
print(data.type.value_counts())#This line talks about transactional page
type = data["type"].value_counts()
transactions = type.index
quantity = type.values
#They show about the distribution of transaction type in pie chart
import plotly.express as px
figure = px.pie(data, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Distribution of Transaction Type")
figure.show()
#correlation = data.corr()
#print(correlation["isFraud"].sort_values(ascending=False))#Now we've to check the correlation between them
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})#It transform isFraud column into No Fraud
correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending=False))
print(data.head())
numeric_data = data.select_dtypes(include=[np.number])
correlation = numeric_data.corr()
print(correlation)
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])#This is used for splitting the data
y = np.array(data[["isFraud"]])
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)#This is for training the ml model
print(model.score(xtest, ytest))
features = np.array([[4, 9000.60, 9000.60, 0.0]])#features=[type,amount,oldbalanceOrg,newbalanceOrig]
print(model.predict(features))
#So at last this model used the decision tree classifier model to classify the Online fraud detection
#And this dedision tree classifier is a trained model.
#We evaluate the performance of the model.
#we return the model score in float and we make a prediction using the trained model.
