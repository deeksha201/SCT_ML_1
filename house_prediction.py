import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("train.csv")

data = data.dropna(subset=['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice'])

X = data.loc[:, ['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data.loc[:, 'SalePrice']

print("Data shape:", data.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model Score:", model.score(X_test, y_test))
plt.figure(figsize=(8,6))

plt.scatter(data['GrLivArea'], data['SalePrice'])
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.title("GrLivArea vs House Price")
plt.legend()
plt.show()
