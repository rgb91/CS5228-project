"""
Created by Sanjay at 10/16/2018

Feature: Enter feature name here
Enter feature description here
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# df = pd.read_csv(r'C:\Users\Sanjay Saha\CS5228-project\data\train_v2.csv')
# df = pd.read_csv(r'C:\Users\Sanjay Saha\CS5228-project\data\trained.word2vec', sep=" ", header=None)
# df = df.drop(df.columns[0], axis=1)
#
# df_for_category = pd.read_csv()
# print(df.shape)

X_300 = np.load(r'C:\Users\Sanjay Saha\CS5228-project\data\newdata\word2vec_300.npy')
X_600 = np.load(r'C:\Users\Sanjay Saha\CS5228-project\data\newdata\word2vec_600.npy')
X_1000 = np.load(r'C:\Users\Sanjay Saha\CS5228-project\data\newdata\word2vec_1000.npy')

df = pd.read_csv(r'C:\Users\Sanjay Saha\CS5228-project\data\train_v_category', sep='\t', header=None)
df = df.drop(df.columns[0], axis=1)
y = np.array(df)
# print(df[0].value_counts())

# pd.value_counts(df['category']).plot.bar()
# plt.title('Article Class Histogram')
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.show()

sm = SMOTE(random_state=2)
X_300, y_300 = sm.fit_sample(X_300, y)
X_600, y_600 = sm.fit_sample(X_600, y)
X_1000, y_1000 = sm.fit_sample(X_1000, y)

np.save(r"C:\Users\Sanjay Saha\CS5228-project\data\smote\y_300", y_300)
np.save(r"C:\Users\Sanjay Saha\CS5228-project\data\smote\y_600", y_600)
np.save(r"C:\Users\Sanjay Saha\CS5228-project\data\smote\y_1000", y_1000)