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

# %%
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# %%
with zipfile.ZipFile('Social_Networks.csv', 'r') as zip_ref:
    zip_ref.extractall()

# %%
df = pd.read_csv('Social_Network_Ads.csv')

# %%
df.head()

# %%
df.isnull().sum()

# %%
X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].values
df.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# %%
y_pred = model.predict(X_test)

# %%
cm = confusion_matrix(y_test, y_pred)
print(cm)

# %%
tn, fp, fn, tp = cm.ravel()
print(f'TN: {tn}')
print(f'FP: {fp}')
print(f'FN: {fn}')
print(f'TP: {tp}')

# %%
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
error_rate = 1- accuracy

print(accuracy)
print(precision)
print(recall)
print(error_rate)

# %%
