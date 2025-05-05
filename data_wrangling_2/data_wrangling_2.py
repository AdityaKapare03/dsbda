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
import  numpy as np
import seaborn as sns

# %%
marks = pd.read_csv('student_marks.csv')

# %%
marks.isnull()

# %%
marks.isnull().sum()

# %%
marks.fillna({'Subject 3':marks['Subject 3'].mean()}, inplace = True)

# %%
marks.dropna(inplace = True)

# %%
numeric_columns = marks.select_dtypes(include=['number']).columns
marks[numeric_columns] = marks[numeric_columns].clip(lower=0)

# %%
marks.head()

# %%
marks.drop_duplicates(inplace=True)

# %%
marks.shape

# %%
numeric_columns = numeric_columns.drop('Roll No')

# %%
import matplotlib.pyplot as plt

# %%
marks = marks[marks['Attendance']>0]

# %%
plt.figure(figsize = (10, 8))
sns.boxplot(data = marks[numeric_columns])
# plt.xticks(rotation=45)
plt.title('To find outliers')
plt.show()

# %%
from scipy import stats

def outliers_iqr(data, column, threshold = 1.5):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3-q1
    lower_bound = q1 - iqr*threshold
    upper_bound = q3 + iqr*threshold
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    if not outliers.empty:
        print(f'Outliers in {column}:')
        print(outliers[['Roll No', 'Name', column]])
    else:
        print('No outliers found!')

def outlier_z(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

for col in numeric_columns:
    outliers_iqr(marks, col)

# %%
for col in numeric_columns:
    lower = marks[col].quantile(0.01)
    upper = marks[col].quantile(0.99)
    marks[col] = marks[col].clip(lower, upper)

# %%
from scipy.stats import skew
for col in numeric_columns:
    original_skew = skew(marks[col])
    plt.figure(figsize=(6,3))
    # plt.subplot(1,2,1)
    sns.histplot(data=marks[col], kde=True)
    plt.title(f'Skewness: {original_skew:.2f}')

# %%
marks['Transformed_attendance'] = np.sqrt(marks['Attendance'])

# %%
marks.head()

# %%
new_skew = skew(marks['Transformed_attendance'])
plt.figure(figsize=(6,3))
plt.xticks(rotation=45)
plt.title(f'Attendace Skewness: {new_skew:.2f}')
sns.histplot(marks['Transformed_attendance'], kde=True)

# %%
from scipy.stats import norm
plt.figure(figsize=(10,6))

for col in numeric_columns:
    data = marks[col].dropna()  # Handle missing values
    mean, std = data.mean(), data.std()
    
    # Create normal distribution curve
    x = np.linspace(data.min(), data.max(), 100)
    y = norm.pdf(x, mean, std)
    
    # Plot with proper labels
    sns.histplot(data, kde=True, stat='density', alpha=0.4, label=f'{col} (Actual)')
    plt.plot(x, y, '--', lw=2, label=f'{col} (Normal)')

plt.title('Actual Distributions vs Normal Distribution')
plt.legend(bbox_to_anchor=(1.05, 1))  # Now has labels to display
plt.tight_layout()
plt.show()

# %%
print(numeric_columns)

# %%
numeric_columns = numeric_columns.drop('Attendance')

# %%
