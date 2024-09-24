import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


ship_tr=pd.read_csv("../data/titanic/train.csv")
ship_test=pd.read_csv("../data/titanic/test.csv")
ship_df=pd.read_csv("../data/titanic/sample_submission.csv")

ship_tr.shape
ship_tr.head()
ship_test.head()

ship_tr.info()
ship_tr.isnull().sum()
ship_test.isnull().sum()



# baseline(전처리, 결측값)
quantitative = ship_tr.select_dtypes(include = [int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]
for col in quant_selected:
    ship_tr[col].fillna(ship_tr[col].mean(), inplace=True)

qualitative = ship_tr.select_dtypes(include = [object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]
for col in qual_selected:
    ship_tr[col].fillna(ship_tr[col].mode()[0], inplace=True)

quantitative = ship_test.select_dtypes(include = [int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]
for col in quant_selected:
    ship_test[col].fillna(ship_test[col].mean(), inplace=True)

qualitative = ship_test.select_dtypes(include = [object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]
for col in qual_selected:
    ship_test[col].fillna(ship_test[col].mode()[0], inplace=True)


# 통합 df 만들기 + 더미코딩
# mush_test.select_dtypes(include=[int, float])
train_n=len(ship_tr)

df = pd.concat([ship_tr, ship_test], ignore_index=True)
df=df.drop("PassengerId", axis=1)
df=df.drop("Name", axis=1)
# df.info()
col = df.select_dtypes(include=[object]).columns
col = col[:-1]
df = pd.get_dummies(
    df,
    columns= col,
    drop_first=True
    )
df

train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

## train
train_x=train_df.drop("Transported", axis=1)
train_y=train_df["Transported"]
train_y = train_y.astype("bool")

## test
test_x=test_df.drop("Transported", axis=1)

# 데이터 스케일링 (표준화)
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# 로지스틱회귀 모델 사용
log_reg = LogisticRegression(random_state = 42)
log_reg.fit(train_x_scaled, train_y)

# 6. 학습한 모델로 예측
y_pred = log_reg.predict(train_x_scaled)


# 첫 번째 층 모델 학습 (RandomForest, XGBoost, LightGBM)
rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [200],
    'max_depth': [7],
    'min_samples_split': [20],
    'min_samples_leaf': [5]
}
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, scoring='accuracy', cv=5)
grid_search_rf.fit(train_x_scaled, train_y)
best_rf_model = grid_search_rf.best_estimator_

xgb_model = XGBClassifier(random_state=42)
param_grid_xgb = {
    'learning_rate': [0.1],
    'n_estimators': [100],
    'max_depth': [7]
}
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, scoring='accuracy', cv=5)
grid_search_xgb.fit(train_x_scaled, train_y)
best_xgb_model = grid_search_xgb.best_estimator_

lgbm_model = LGBMClassifier(random_state=42)
param_grid_lgbm = {
    'learning_rate': [0.05],
    'n_estimators': [200],
    'max_depth': [7]
}
grid_search_lgbm = GridSearchCV(estimator=lgbm_model, param_grid=param_grid_lgbm, scoring='accuracy', cv=5)
grid_search_lgbm.fit(train_x_scaled, train_y)
best_lgbm_model = grid_search_lgbm.best_estimator_

# 첫 번째 층 모델 예측값 생성
y1_hat = best_rf_model.predict(train_x_scaled)
y2_hat = best_xgb_model.predict(train_x_scaled)
y3_hat = best_lgbm_model.predict(train_x_scaled)
y4_hat = log_reg.predict(train_x_scaled)

train_x_stack_1 = pd.DataFrame({'y1': y1_hat,
                                 'y2': y2_hat,
                                   'y3': y3_hat,
                                     'y4': y4_hat})

# 두 번째 층 모델 학습 (GradientBoostingClassifier)
gb_model = GradientBoostingClassifier(random_state=42)
param_grid_gb = {
    'learning_rate': [0.01],
    'n_estimators': [100],
    'max_depth': [5]
}
grid_search_gb = GridSearchCV(estimator=gb_model, param_grid=param_grid_gb, scoring='accuracy', cv=5)
grid_search_gb.fit(train_x_stack_1, train_y)
best_gb_model = grid_search_gb.best_estimator_

# 테스트 데이터 예측
pred_y_rf = best_rf_model.predict(test_x_scaled)
pred_y_xgb = best_xgb_model.predict(test_x_scaled)
pred_y_lgbm = best_lgbm_model.predict(test_x_scaled)
pred_y_logi = log_reg.predict(test_x_scaled)

test_x_stack_1 = pd.DataFrame({'y1': pred_y_rf, 'y2': pred_y_xgb, 'y3': pred_y_lgbm, 'y4': pred_y_logi})

pred_y_gb = best_gb_model.predict(test_x_stack_1)

# 결과 저장
ship_df["Transported"] = pred_y_gb
ship_df.to_csv("../data/titanic/sample_submission_base.csv", index=False)



