import pandas as pd
import numpy as np
import seaborn as sns

iris = sns.load_dataset('iris')
iris.head()

iris.info()

iris['species'].unique()

summary_stats = iris.groupby('species')['sepal_length'].agg(['mean', 'median', 'max', 'min', 'std']).reset_index()

summary_stats

grouped_stats = iris.groupby('species').describe()
grouped_stats.T

grouped_qualitative = iris.groupby('species')['sepal_length'].describe()
grouped_qualitative

set = iris['species'].unique()

for species in set:
    print(f'\n{'='*50}')
    print(f'Species: {species}')
    print('='*50)
    species_data = iris[iris['species']==species]
    display(species_data.describe(percentiles=[.25, .5, .75]))

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for species in set:
    print(f'\n{'='*50}')
    print(f'Species: {species}')
    print('='*50)
    species_data = iris[iris['species'] == species]
    for feature in features:
        print(f'\nFeature: {feature}')
        data = species_data[feature]

        print(f'Count: {len(data)}')
        print(f'Mean: {data.mean():.2f}')
        print(f'Median: {data.median():.2f}')
        print(f'Min: {data.min():.2f}')
        print(f'Max: {data.max():.2f}')
        print(f'Std: {data.std():.2f}')
        print(f'q1: {data.quantile(0.25):.2f}')
        print(f'q3: {data.quantile(0.75):.2f}')
