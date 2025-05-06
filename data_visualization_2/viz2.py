import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

plt.figure(figsize=(10, 5))
sns.boxplot(data=titanic, x='sex', y='age', hue='survived', palette='Set2')
plt.show()

sns.histplot(data=titanic, x='sex', multiple='dodge', hue='survived')
plt.show()
