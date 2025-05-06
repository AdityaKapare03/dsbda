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

titanic = sns.load_dataset('titanic')

# %%
titanic.head()

# %%
sns.pairplot(data = titanic, hue='survived', vars=['pclass', 'age', 'fare'], palette = 'Set1')
plt.suptitle('Basic patterns in titanic dataset', y=1.02)
plt.show()

# %%
plt.figure(figsize=(8,5))
sns.histplot(data=titanic, kde = True, bins=50, x='fare')
plt.xlabel('fare')
plt.ylabel('Number of Passengers')
plt.show()

# %%
