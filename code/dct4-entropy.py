
# 펭귄 데이터 부리길이 예측 모형 만들기
# 엘라스틱 넷 & 디시젼트리 회귀모델 사용
# 모든 변수 자유롭게 사용!
# 종속변수: bill_length_mm

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.preprocessing import OneHotEncoder

penguins = load_penguins()
penguins.head()

df_X = penguins.drop("species", axis = 1)
df_X = df_X[["bill_length_mm", "bill_depth_mm"]]
y = penguins[["species"]]

## Nan 채우기
quantitative = penguins.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    penguins[col].fillna(penguins[col].mean(), inplace=True)
penguins[quant_selected].isna().sum()

## 범주형 채우기
qualitative = penguins.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    penguins[col].fillna(penguins[col].mode()[0], inplace=True)
penguins[qual_selected].isna().sum()

df = penguins
df = pd.get_dummies(
    df,
    columns = df.select_dtypes(include=[object]).columns,
    drop_first = True
)
df

x=df.drop("species", axis=1)
y=df[['species']]
x
y

# 모델 생성
from sklearn.tree import DecisionTreeClassifier

## 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV
model = DecisionTreeClassifier(criterion = "entropy", random_state=42, max_depth = 2)

model.fit(df_X, y)
model.predict(df_X)


param_grid={
    'max_depth': np.arange(7, 20, 1),
    'min_samples_split': np.arange(10, 30, 1)
}

grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5
)

grid_search.fit(x,y)

grid_search.best_params_ #8, 22
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

model = DecisionTreeClassifier(random_state=42,
                              max_depth=2,
                              min_samples_split=22)
model.fit(x,y)

from sklearn import tree
tree.plot_tree(model)


from sklearn.metrics import confusion_matrix
import numpy as np
# 아델리: 'A'
# 친스트랩(아델리 아닌것): 'C'
y_true = np.array(['A', 'A', 'C', 'A', 'C', 'C', 'C'])
y_pred = np.array(['A', 'C', 'A', 'A', 'A', 'C', 'C'])
y_pred2 = np.array(['C', 'A', 'A', 'A', 'C', 'C', 'C'])

conf_mat = confusion_matrix(y_true=y_true, 
                 y_pred=y_pred,
                 labels=["A", "C"])

conf_mat

from sklearn.metrics import ConfusionMatrixDisplay

p = ConfusionMatrixDisplay(confusion_matrix = conf_mat,
                           display_labels = ("Adelie", "Chinstrap"))
p.plot(cmap = "Blues")

conf_mat = confusion_matrix(y_true=y_true, 
                 y_pred=y_pred2,
                 labels=["A", "C"])

conf_mat

from sklearn.metrics import ConfusionMatrixDisplay

p = ConfusionMatrixDisplay(confusion_matrix = conf_mat,
                           display_labels = ("Adelie", "Chinstrap"))
p.plot(cmap = "Blues")

