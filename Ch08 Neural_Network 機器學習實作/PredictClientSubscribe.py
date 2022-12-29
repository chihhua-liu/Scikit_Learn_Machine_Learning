#Multi layer Perceptron (MLP) Models on Real World Banking Data(https://becominghuman.ai/multi-layer-perceptron-mlp-models-on-real-world-banking-data-f6dd3d7e998f)

import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

df = pd.read_csv(r'Datasets\bank.csv', sep=';')
print(df.head())
print(df.info())

#df.to_csv("bank.csv")

encodeColumns=["job", "marital", "education", "housing", "loan", "contact", "poutcome", "y"]
df.loc[:, encodeColumns]=df.loc[:, encodeColumns].apply(LabelEncoder().fit_transform)
df=df.drop(['month','default'], axis=1)
df.rename(columns={"y": "subscribed"}, inplace=True)
print(df.head())

X=df.drop('subscribed',axis=1)
Y= df['subscribed']

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=101)

scaler = StandardScaler()

# Fit the Training & Test Data 
scaler.fit(X_Train)

X_Train = scaler.transform(X_Train)
X_Test = scaler.transform(X_Test)

model = MLPClassifier(hidden_layer_sizes=(11,8,5), activation="relu", max_iter=1000)
model.fit(X_Train, Y_Train)

predictions =model.predict(X_Test)
print(confusion_matrix(Y_Test, predictions))
print(classification_report(Y_Test, predictions))
print(accuracy_score(Y_Test,predictions))

scores = cross_val_score(model, X, Y, cv=10)
print(scores)
print(scores.mean())
