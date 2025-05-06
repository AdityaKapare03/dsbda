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
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import seaborn

# %%
iris = seaborn.load_dataset('iris')
iris.head()

# %%
X = iris.drop('species', axis = 1)
y = iris['species']

# %%
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.25, random_state = 42)

# %%
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('sample predictions', y_pred[::-5])

# %%
cm = confusion_matrix(y_test, y_pred)
print(cm)


# %%
def class_metrics(cm, index):
    tp = cm[index, index]
    fp = cm[:, index].sum() - tp
    fn = cm[index, :].sum() - tp
    tn = cm.sum() - tp - fp - fn
    return tp, fp, fn, tn
    


# %%
classes = np.unique(y_test)

# %%
print('Matrix per class')
for i, class_name in enumerate(classes):
    tp, fp, fn, tn = class_metrics(cm, i)
    accuracy = (tp+tn)/(tp+tn+fn+fp)
    precision = tp / (tp+fp) if (tp+fp)>0 else 0
    recall = tp / (tp+fn) if (tp+fn)>0 else 0

    print(f'Class {class_name}')
    print(f'TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')

# %%
overall_accuracy = accuracy_score(y_test, y_pred)
print(f'Overall accuracy: {overall_accuracy}')

# %%
print('Classification Report: \n', classification_report(y_test, y_pred))

# %%
