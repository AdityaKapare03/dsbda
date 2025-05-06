import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

iris = sns.load_dataset('iris')

iris.info()

features = iris.select_dtypes(include=['number']).columns

for feature in features:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=iris, x=feature, hue='species', palette='Set1')
    plt.show()

for feature in features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=iris, x=feature, hue='species', palette='Set1')
    plt.show()
