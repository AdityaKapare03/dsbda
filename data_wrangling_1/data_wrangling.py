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
import pandas as pd
import numpy as np
import seaborn as sns

# %%
df = sns.load_dataset('titanic')

# %%
df.info()

# %%
df.head()

# %%
print("missing values:\n", df.isnull().sum())

# %%
print(df['class'].unique())

# %%
class_mapping = {'First':1, 'Second':2, 'Third':3}
df['class_num'] = df['class'].map(class_mapping)

# %%
df.info()

# %%
df.describe()

# %%
df.shape

# %%
df.dtypes

# %%
df['class_num'] = pd.to_numeric(df['class_num'], errors = 'coerce')

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 3))
sns.countplot(data=df, x='embark_town', hue='class_num', palette='viridis')
plt.title('Graph')
plt.xlabel('Embark Town')
plt.ylabel('Count')
plt.show()

# %%
df = df.drop('deck', axis = 1)

# %%
plt.figure(figsize=(4, 2))
sns.countplot(data=df, x='embark_town')
plt.title('Graph')
plt.xlabel('Embark Town')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(6, 3))
sns.countplot(data=df, x='embark_town', hue='alive', palette='viridis')
plt.title('Graph')
plt.xlabel('Embark Town')
plt.ylabel('Count')
plt.legend(title='Survived?')
plt.show()

# %%
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='class_num', hue='alive', palette='viridis')
plt.title('Graph')
plt.xlabel('class_num')
plt.ylabel('Count')
plt.show()

# %%
df['alive'] = pd.get_dummies(df['alive'], drop_first = True)

# %%
df.info()

# %%
dead = df[df['alive']==0]

# %%
alives = df[df['alive']==1]

# %%
dead.info()

# %%
plt.figure(figsize=(5, 5))
sns.countplot(data=dead, x='embarked', hue='class', palette='viridis')
plt.title('Dead graph')
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.legend(title='Passenger Class')
plt.show()

# %%
death_counts = alives.groupby(['embarked', 'class'], observed=True).size().unstack()
death_counts.plot(kind='bar', stacked=True, figsize=(6, 4), colormap = 'viridis')
plt.title('Alive graph')
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.legend(title='Class')
plt.show()

# %%
