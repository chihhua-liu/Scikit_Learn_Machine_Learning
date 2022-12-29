#Your Guide to Linear Regression Models(https://www.kdnuggets.com/2020/10/guide-linear-regression-models.html)
#Linear Regression in Python(https://medium.com/@harishreddyp98/linear-regression-in-python-c164149b93ab)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import statsmodels.api as sm

df = pd.read_csv(r"Datasets\Advertising.csv")
print(df.head())

df = df.drop(['Unnamed: 0'], axis=1)
print(df.info())

sns.pairplot(df)
plt.show()

mask = np.tril(df.corr())
sns.heatmap(df.corr(), fmt='.1g', annot=True, cmap= 'cool', mask=mask)
plt.show()

X = df.drop(['sales'], axis=1)
Y = df['sales']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

model = LinearRegression()
model.fit(X_train, Y_train)

print(model.intercept_)

coeff_df = pd.DataFrame(model.coef_, X.columns, columns =['Coefficient'])
print(coeff_df)

#驗証模型
y_pred = model.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
print('R Squared Score is:', r2_score(Y_test, y_pred))  # higher R² indicates a better fit for the model

#interpret and improve
#X2 = sm.add_constant(X_train)
#model_stats = sm.OLS(Y_train.values.reshape(-1,1), X2).fit()
#print(model_stats.summary())

#計算 50 for TV, 30 for radio and 10的收益
example = [50, 30, 10]  
output = model.intercept_ + sum(example*model.coef_)
print(f"Estimate Sales:{output}")

example = [30, 50, 10]  
output = model.intercept_ + sum(example*model.coef_)
print(f"Estimate Sales:{output}")

example = [20, 50, 20]  
output = model.intercept_ + sum(example*model.coef_)
print(f"Estimate Sales:{output}")