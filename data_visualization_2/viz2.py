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
import seaborn as sns
import matplotlib.pyplot as plt

# %%
titanic = sns.load_dataset('titanic')
titanic.head()

# %%
plt.figure(figsize=(10,5))
sns.boxplot(data=titanic, x='sex', y='age', hue='survived', palette='Set2')
plt.show()

# %%
sns.histplot(data=titanic, x='sex', multiple='dodge', hue='survived')

# %%
