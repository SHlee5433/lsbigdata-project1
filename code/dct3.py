# 펭귄 데이터 부리길이 예측 모형 만들기
# 엘라스틱 넷 & 디시젼트리 회귀모델 사용
# 모든 변수 자유롭게 사용!

import numpy as np
import pandas as pd
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()
penguins.info()
penguins.isna().sum()

## 숫자형 채우기
quantitative = penguins.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    penguins[col].fillna(penguins[col].mean(), inplace=True)
penguins[quant_selected].isna().sum()

## 범주형 채우기
# 성별 최빈값으로 채우기
# penguins["sex"].mode()[0]으로 작성해야 함
# penguins["sex"].mode()는 함수 결과가 리스트로 나와서 결측값 대체를 못함

qualitative = penguins.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    penguins[col].fillna(penguins["sex"].mode()[0], inplace=True)
penguins[qual_selected].isna().sum()

# 더미코딩
df = penguins
df = pd.get_dummies(
    df,
    columns = df.select_dtypes(include=[object]).columns,
    drop_first = True
)
df

df_x = df.drop("bill_length_mm", axis=1)
df_y = df["bill_length_mm"]


from sklearn.linear_model import ElasticNet
model = ElasticNet()

param_grid = {
    'alpha': np.arange(0.01, 1, 0.1),  # 더 넓은 범위 설정
    'l1_ratio': np.arange(0.9, 1, 0.01)   # 0 ~ 1 사이에서 탐색
}


from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=model,
    param_grid = param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(df_x, df_y)

grid_search.best_params_ # alpha = 0.21 , l1-ratio = 0.99
grid_search.cv_results_
-1 * grid_search.best_score_
best_model = grid_search.best_estimator_



# 디시젼트리 회귀모델

from sklearn.tree import DecisionTreeRegressor


decision = DecisionTreeRegressor(random_state=42)
                               # 최적의 성능을 낼 수 있는 하이퍼 파라미터가 될 수 있다.

param_grid = {
    'max_depth': np.arange(1, 10, 1),  # 더 넓은 범위 설정
    'min_samples_split': np.arange(20, 25, 1)   # 0 ~ 1 사이에서 탐색
}


from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=decision,
    param_grid = param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)


grid_search.fit(df_x, df_y)

grid_search.best_params_ # alpha = 8 , l1-ratio = 22
grid_search.cv_results_
-1 * grid_search.best_score_
best_model = grid_search.best_estimator_

model = DecisionTreeRegressor(random_state=42,
                              max_depth=2,
                              min_samples_split=22)
model.fit(df_x, df_y)

from sklearn import tree
tree.plot_tree(model)
