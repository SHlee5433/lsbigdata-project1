import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# import os
# cwd = os.getcwd()

# 데이터 불러오기
house_train = pd.read_csv("../data/houseprice/train.csv")
house_test = pd.read_csv("../data/houseprice/test.csv")
sub_df = pd.read_csv("../data/houseprice/sample_submission.csv")

# 결측치 처리
## 숫자형 변수
quantitative_train = house_train.select_dtypes(include=[int, float])
quant_selected_train = quantitative_train.columns[quantitative_train.isna().sum() > 0]
for col in quant_selected_train:
    house_train[col].fillna(house_train[col].mean(), inplace=True)

quantitative_test = house_test.select_dtypes(include=[int, float])
quant_selected_test = quantitative_test.columns[quantitative_test.isna().sum() > 0]
for col in quant_selected_test:
    house_test[col].fillna(house_train[col].mean(), inplace=True)

## 범주형 변수
qualitative_train = house_train.select_dtypes(include=[object])
qual_selected_train = qualitative_train.columns[qualitative_train.isna().sum() > 0]
for col in qual_selected_train:
    house_train[col].fillna("unknown", inplace=True)

qualitative_test = house_test.select_dtypes(include=[object])
qual_selected_test = qualitative_test.columns[qualitative_test.isna().sum() > 0]
for col in qual_selected_test:
    mode_value = house_train[col].mode()[0]
    house_test[col].fillna(mode_value, inplace=True)

# 통합 데이터프레임 생성 및 더미코딩
df = pd.concat([house_train, house_test], ignore_index=True)
df = pd.get_dummies(df, columns=df.select_dtypes(include=[object]).columns, drop_first=True)

# train / test 데이터 분리
train_n = len(house_train)
train_df = df.iloc[:train_n, :]
test_df = df.iloc[train_n:, :]

# 이상치 제거
train_df = train_df.query("GrLivArea <= 4500")

# X, y 데이터 분리
train_x = train_df.drop("SalePrice", axis=1)
train_y = train_df["SalePrice"]
test_x = test_df.drop("SalePrice", axis=1)

# 최적 하이퍼파라미터를 사용한 모델 설정
# 1. ElasticNet
pipe_eln = Pipeline([
    ('scaler', StandardScaler()),
    ('elasticnet', ElasticNet(alpha=0.1, l1_ratio=0.5))
])

# 2. Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5)

# 3. Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, subsample=0.8)

# 스태킹 모델 구성
stacking_model = StackingRegressor(
    estimators=[
        ('elasticnet', pipe_eln),
        ('rf', rf_model),
        ('gb', gb_model)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

# 스태킹 모델 학습
stacking_model.fit(train_x, train_y)

# 테스트 데이터 예측
stacking_preds = stacking_model.predict(test_x)

# 결과 저장
sub_df['SalePrice'] = stacking_preds
sub_df.to_csv("../data/houseprice/sample_submission10.csv", index=False)
