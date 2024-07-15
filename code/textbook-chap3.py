# 교재 63 페이지

# seaborn 패키지 설치
# !pip install seaborn

import seaborn as sns
import matplotlib.pyplot as plt

var=["a", "a", "b", "c"]
var

sns.countplot(x=var)
plt.show()
plt.clf()

df = sns.load_dataset("titanic")
sns.countplot(data = df, x = "sex")
sns.countplot(data = df, x = "sex", hue="sex")
plt.show()

?sns.countplot
sns.countplot(data=df, x="class")
sns.countplot(data=df, x="class", hue="alive")
sns.countplot(data=df,
              x="survived",
              y="pclass")
plt.show()

# !pip install scikit-learn
import sklearn.metrics
# sklearn.metrics.accuracy_score()

from sklearn import metrics
# metrics.accuracy_score()

import sklearn.metrics as met
# met.accuracy_score()



#!/usr/bin/env python
import os
print(os.getcwd())

import pandas as pd

df_exam = pd.read_csv('exam.csv')
df_exam
