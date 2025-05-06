import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

titanic.head()

sns.pairplot(data=titanic, hue='survived', vars=['pclass', 'age', 'fare'], palette='Set1')
plt.suptitle('Basic patterns in titanic dataset', y=1.02)
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data=titanic, kde=True, bins=50, x='fare')
plt.xlabel('fare')
plt.ylabel('Number of Passengers')
plt.show()
