# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---



# import all the libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# import dataset
boston = pd.read_csv('housing_data.csv')


# Exploratory data analysis (EDA)
boston.head()
boston.shape
boston.isnull().sum()


# select the numeric columns
features = boston.select_dtypes(include = ['number']).columns
print(features)


# handling null values
boston[features]=boston[features].fillna(boston[features].median())


# checking again
boston.isnull().sum()

#draw the heatmap to bring out the most prominent features
plt.figure(figsize=(9,9))
sns.heatmap(data=boston.corr().round(2), annot=True, square=True, cmap='coolwarm', linewidth=0.2)

# result of heatmap
prime_features = ['LSTAT', 'RM', 'DIS', 'NOX', 'TAX', 'MEDV']

#create new dataset
df = boston[prime_features]
df.shape

# optional
sns.pairplot(df)

for feature in prime_features:
    plt.figure(figsize=(6,3))
    plt.subplot(121)
    plt.title(f'{feature} Distribution')
    sns.histplot(df[feature], kde=True)
    plt.subplot(122)
    plt.title(f'{feature} Boxplot')
    sns.boxplot(df[feature])
    plt.tight_layout()
    plt.show()
# optional ended

# outliers removal start
def outliers(data, threshold = 1.5):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr*threshold
    upper = q3 + iqr*threshold
    return (data<lower)|(data>upper)

count = df.apply(lambda x: outliers(x).sum())
print(count)

# remove outliers from df and boston(not needed)
outlier_cols = ['LSTAT', 'MEDV', 'RM']
outlier_mask = df[outlier_cols].apply(lambda x: outliers(x)).any(axis=1)
df = df[~outlier_mask]
df.shape
#optional
outlier_mask = boston[outlier_cols].apply(lambda x: outliers(x)).any(axis=1)
boston = boston[~outlier_mask]
boston.shape

#declare the variables to model training
X = df.drop('MEDV', axis = 1) #consider all except the MEDV
y = df['MEDV']

# do it or don't
print(f'X Shape: {X.shape}, Y Shape: {y.shape}')

# splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(f'Train shapes: X Shape: {X_train.shape}, Y Shape: {y_train.shape}')
print(f'Test shapes: X Shape: {X_test.shape}, Y Shape: {y_test.shape}')

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# predict the results
y_pred = model.predict(X_test_scaled)

#test the errors
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f'R2 score: {r2:.2f}') #closer to 1 is better
print(f'Mean absolute error: {mae:.2f}') #lower the beter
print(f'Mean squared error: {mse:.2f}') #lower the better
print(f'Root mean squared error: {rmse:.2f}')

# check for the comparison between model and the actual prices
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha = 0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--') 
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual VS Predicted House Prices')
plt.show()

# check for the efficiency and the correctenss of the graph
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.residplot(x=y_pred, y=residuals, lowess = True, line_kws={'color':'red'})
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# importance of features
coefficients = pd.DataFrame({'Feature':X.columns, 'Coefficients':model.coef_}).sort_values(by='Coefficients', ascending=False)

plt.figure(figsize = (8, 5))
sns.barplot(x='Coefficients', y='Feature', data = coefficients)
plt.title('Feature Importance')
plt.show()
