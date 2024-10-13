import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.metrics import roc_auc_score, precision_score,recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


# 데이터 불러오기
df_raw = pd.read_csv("../data/bigdata/data_week2.csv", encoding ="cp949")

df = df_raw.copy()

df.columns = ['num', 'datetime', 'target', 'temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df["datetime"] = pd.to_datetime(df["datetime"])
df['month'] = df.datetime.dt.month                    # 월(숫자)
df['day'] = df.datetime.dt.day                        # 일(숫자)
df['hour'] = df.datetime.dt.hour                      # 시(숫자)
df['weekday'] = df.datetime.dt.weekday                # 요일(숫자)
df['dayofyear'] = df.datetime.dt.dayofyear

df['weekday'].unique() # 0, 1, 2, 3, 4, 5, 6 (0 = 월요일)

df["target"].describe()
df["target"].hist()

df["temp"].describe()
df["temp"].hist()

df["wind"].describe()
df["wind"].hist()

df["humid"].describe()
df["humid"].hist()

df["rain"].describe()
df["rain"].hist()

df.drop("datetime",axis=1,inplace=True)

X = df.drop(columns=['target'])
y = df['target']

X_train, X_valid, y_train, y_valid = temporal_train_test_split(X, y, test_size = 168) # 24시간*7일 = 168


##### xgboost

xgb_model = XGBRegressor(random_state=42, eval_metric = "rmse")
xgb_model.fit(X_train, y_train)


xgb_pred = xgb_model.predict(X_valid)

# XGB모델의 RMSE값
xgb_rmse = np.sqrt(mean_squared_error(y_valid, xgb_pred))
print(xgb_rmse) # 235(기본값)

# XGB모델의 SMAPE값
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

smape(y_valid, xgb_pred) # 4.83(기본값)


#####lgbm
lgb_model = LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)

lgb_pred = lgb_model.predict(X_valid)

# LGBM모델의 RMSE값
lgb_rmse = np.sqrt(mean_squared_error(y_valid, lgb_pred))
print(lgb_rmse) # 322(기본값)

# LGBM모델의 SMAPE값
smape(y_valid, lgb_pred) # 7.37(기본값)

#####catboost
cat_model = CatBoostRegressor(random_state=42, verbose=0)  # verbose=0: 학습 로그 출력하지 않음
cat_model.fit(X_train, y_train)

cat_pred = cat_model.predict(X_valid)

# cat모델의 RMSE값
cat_rmse = np.sqrt(mean_squared_error(y_valid, cat_pred))
print(cat_rmse) # 187(기본값)

# cat모델의 SMAPE값

smape(y_valid, cat_pred) # 4.21(기본값)



# randomforest모델 RMSE값

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 모델 훈련
rf_model.fit(X_train, y_train)

# 예측
y_pred = rf_model.predict(X_valid)

# RMSE 계산
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f'RMSE: {rmse}') # 227

smape(y_valid, y_pred) # 5.13



sum(df["target"] == 0)