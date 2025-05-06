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
import pandas as pd
import numpy as np

iris = sns.load_dataset('iris')

# %%
iris.info()

# %%
features = iris.select_dtypes(include=['number']).columns

# %%
for feature in features:
    plt.figure(figsize=(10,5))
    sns.histplot(data=iris, x=feature, hue='species', palette='Set1')
    plt.show()

# %%
for feature in features:
    plt.figure(figsize=(10,5))
    sns.boxplot(data=iris, x=feature, hue='species', palette='Set1')
    plt.show()
