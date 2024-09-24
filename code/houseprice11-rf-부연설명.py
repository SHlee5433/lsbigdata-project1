# 블랜더를 돌리면 가중치를 찾아준 모델 예측값들에 또 가중치를 줘서 합침
# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 워킹 디렉토리 설정
# import os
# cwd=os.getcwd()
# parent_dir = os.path.dirname(cwd)
# os.chdir(parent_dir)

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

## NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
house_train.isna().sum()
house_test.isna().sum()

## 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()


# test 데이터 채우기
## 숫자형 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_test[col].fillna(house_train[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_test.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_test[col].fillna("unknown", inplace=True)
house_test[qual_selected].isna().sum()


house_train.shape
house_test.shape
train_n=len(house_train)


# test 데이터 채우기
## 숫자형 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_test[col].fillna(house_train[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_test.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    mode_value = house_train[col].mode()[0]  # 최빈값 계산
    house_test[col].fillna(mode_value, inplace=True) 

house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
# house_test.select_dtypes(include=[int, float])

df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]


## 이상치 탐색
# train_df=train_df.query("GrLivArea <= 4500")


## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

train_x.shape[0]
btstrap_index1 = np.random.choice(np.arange(1458), 1000, replace = True)
bts_train_x1 = train_x.iloc[btstrap_index1,:]
bts_train_y1 = np.array(train_y)[btstrap_index1]

train_x.shape[0]
btstrap_index2 = np.random.choice(np.arange(1458), 1458, replace = True)
bts_train_x2 = train_x.iloc[btstrap_index2,:]
bts_train_y2 = np.array(train_y)[btstrap_index2]


model = DecisionTreeRegressor(random_state = 42)
param_grid={
    'max_depth': np.arange(7, 20, 1),
    'min_samples_split': np.arange(10, 30, 1)
}
grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(bts_train_x1, bts_train_y1)
grid_search.best_params_
bts_model1 = grid_search.best_estimator_

grid_search.fit(bts_train_x2, bts_train_y2)
grid_search.best_params_
bts_model2 = grid_search.best_estimator_

bts1_y = bts_model1.predict(train_x)
bts2_y = bts_model2.predict(train_x)
(bts1_y + bts2_y)/2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 한결 코드 ~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 블랜더를 돌리면 가중치를 찾아준 모델 예측값들에 또 가중치를 줘서 합침
# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 워킹 디렉토리 설정
# import os
# cwd=os.getcwd()
# parent_dir = os.path.dirname(cwd)
# os.chdir(parent_dir)

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

## NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
house_train.isna().sum()
house_test.isna().sum()

## 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()


# test 데이터 채우기
## 숫자형 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_test[col].fillna(house_train[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_test.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_test[col].fillna("unknown", inplace=True)
house_test[qual_selected].isna().sum()


house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
# house_test.select_dtypes(include=[int, float])

df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

## 이상치 탐색
train_df=train_df.query("GrLivArea <= 4500")

## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)

# -------------------------------------------------------

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

# 부트스트랩 - 서브 데이터 셋
train_x.shape[0]
btstrap_index1=np.random.choice(np.arange(1458), 1000, replace=True)
bts_train_x1=train_x.iloc[btstrap_index1,:]
bts_train_y1=np.array(train_y)[btstrap_index1]

btstrap_index2=np.random.choice(np.arange(1458), 1000, replace=True)
bts_train_x2=train_x.iloc[btstrap_index2,:]
bts_train_y2=np.array(train_y)[btstrap_index2]

btstrap_index3=np.random.choice(np.arange(1458), 1000, replace=True)
bts_train_x3=train_x.iloc[btstrap_index3,:]
bts_train_y3=np.array(train_y)[btstrap_index3]

btstrap_index4=np.random.choice(np.arange(1458), 1000, replace=True)
bts_train_x4=train_x.iloc[btstrap_index4,:]
bts_train_y4=np.array(train_y)[btstrap_index4]

btstrap_index5=np.random.choice(np.arange(1458), 1000, replace=True)
bts_train_x5=train_x.iloc[btstrap_index5,:]
bts_train_y5=np.array(train_y)[btstrap_index5]

# ----------------------------------------------------------

eln_model= ElasticNet()
rf_model= RandomForestRegressor(n_estimators=100)
# knn_model= KNeighborsRegressor()

# 그리드 서치 for ElasticNet
param_grid={
    'alpha': np.arange(10.0, 100.0, 10),
    'l1_ratio': [0.9, 0.925, 0.95, 0.975, 1.0]
}
grid_search=GridSearchCV(
    estimator=eln_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(bts_train_x1, bts_train_y1)
grid_search.best_params_
bts_eln_model1=grid_search.best_estimator_

grid_search.fit(bts_train_x2, bts_train_y2)
grid_search.best_params_
bts_eln_model2=grid_search.best_estimator_

grid_search.fit(bts_train_x3, bts_train_y3)
grid_search.best_params_
bts_eln_model3=grid_search.best_estimator_

grid_search.fit(bts_train_x4, bts_train_y4)
grid_search.best_params_
bts_eln_model4=grid_search.best_estimator_

grid_search.fit(bts_train_x5, bts_train_y5)
grid_search.best_params_
bts_eln_model5=grid_search.best_estimator_


# 그리드 서치 for RandomForests
param_grid={
    'max_depth': [3, 5, 7],
    'min_samples_split': [20, 10, 5],
    'min_samples_leaf': [5, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None]
}
grid_search=GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(bts_train_x1, bts_train_y1)
bts_rf_model1=grid_search.best_estimator_

grid_search.fit(bts_train_x2, bts_train_y2)
bts_rf_model2=grid_search.best_estimator_

grid_search.fit(bts_train_x3, bts_train_y3)
bts_rf_model3=grid_search.best_estimator_

grid_search.fit(bts_train_x4, bts_train_y4)
bts_rf_model4=grid_search.best_estimator_

grid_search.fit(bts_train_x5, bts_train_y5)
bts_rf_model5=grid_search.best_estimator_


# # 그리드 서치 for knn
# param_grid={
#     'n_neighbors': [3, 5, 7, 9]
# }
# grid_search=GridSearchCV(
#     estimator=knn_model,
#     param_grid=param_grid,
#     scoring='neg_mean_squared_error',
#     cv=5
# )

# grid_search.fit(bts_train_x1, bts_train_y1)
# bts_knn_model1=grid_search.best_estimator_

# grid_search.fit(bts_train_x2, bts_train_y2)
# bts_knn_model2=grid_search.best_estimator_

# grid_search.fit(bts_train_x3, bts_train_y3)
# bts_knn_model3=grid_search.best_estimator_

# grid_search.fit(bts_train_x4, bts_train_y4)
# bts_knn_model4=grid_search.best_estimator_

# grid_search.fit(bts_train_x5, bts_train_y5)
# bts_knn_model5=grid_search.best_estimator_

# ----------------------------------------

# 스택킹
y1_hat= bts_eln_model1.predict(train_x) 
y2_hat= bts_eln_model2.predict(train_x)
y3_hat= bts_eln_model3.predict(train_x)
y4_hat= bts_eln_model4.predict(train_x)
y5_hat= bts_eln_model5.predict(train_x)
y6_hat= bts_rf_model1.predict(train_x)
y7_hat= bts_rf_model2.predict(train_x)
y8_hat= bts_rf_model3.predict(train_x)
y9_hat= bts_rf_model4.predict(train_x)
y10_hat=bts_rf_model5.predict(train_x)
# y11_hat=bts_knn_model1.predict(train_x)
# y12_hat=bts_knn_model2.predict(train_x)
# y13_hat=bts_knn_model3.predict(train_x)
# y14_hat=bts_knn_model4.predict(train_x)
# y15_hat=bts_knn_model5.predict(train_x)

# 'yi' : yi_hat 자동 완성 code
for i in range(1, 16):
    if i < 15:  
        print(f"'y{i}': y{i}_hat,")
    else:  
        print(f"'y{i}': y{i}_hat")

train_x_stack=pd.DataFrame({
    'y1': y1_hat,
    'y2': y2_hat,
    'y3': y3_hat,
    'y4': y4_hat,
    'y5': y5_hat,
    'y6': y6_hat,
    'y7': y7_hat,
    'y8': y8_hat,
    'y9': y9_hat,
    # 'y10': y10_hat,
    # 'y11': y11_hat,
    # 'y12': y12_hat,
    # 'y13': y13_hat,
    # 'y14': y14_hat,
    # 'y15': y15_hat
})


from sklearn.linear_model import Ridge

rg_model = Ridge()
param_grid={
    'alpha': np.arange(0, 10, 0.01)
}
grid_search=GridSearchCV(
    estimator=rg_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(train_x_stack, train_y)
grid_search.best_params_
blander_model=grid_search.best_estimator_

blander_model.coef_
blander_model.intercept_

pred_y1_eln= bts_eln_model1.predict(test_x) 
pred_y2_eln= bts_eln_model2.predict(test_x)
pred_y3_eln= bts_eln_model3.predict(test_x)
pred_y4_eln= bts_eln_model4.predict(test_x)
pred_y5_eln= bts_eln_model5.predict(test_x)
pred_y1_rf = bts_rf_model1.predict(test_x)
pred_y2_rf = bts_rf_model2.predict(test_x)
pred_y3_rf = bts_rf_model3.predict(test_x)
pred_y4_rf = bts_rf_model4.predict(test_x)
pred_y5_rf = bts_rf_model5.predict(test_x)
# pred_y1_knn = bts_knn_model1.predict(test_x)
# pred_y2_knn = bts_knn_model2.predict(test_x)
# pred_y3_knn = bts_knn_model3.predict(test_x)
# pred_y4_knn = bts_knn_model4.predict(test_x)
# pred_y5_knn = bts_knn_model5.predict(test_x)

test_x_stack=pd.DataFrame({
    'y1': pred_y1_eln,
    'y2': pred_y1_eln,
    'y3': pred_y3_eln,
    'y4': pred_y4_eln,
    'y5': pred_y5_eln,
    'y6': pred_y1_rf,
    'y7': pred_y2_rf,
    'y8': pred_y3_rf,
    'y9': pred_y4_rf,
    'y10': pred_y5_rf,
    # 'y11': pred_y1_knn,
    # 'y12': pred_y2_knn,
    # 'y13': pred_y3_knn,
    # 'y14': pred_y4_knn,
    # 'y15': pred_y5_knn
})

pred_y=blander_model.predict(test_x_stack)

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# # csv 파일로 내보내기
# sub_df.to_csv("./data/houseprice/sample_submission28.csv", index=False)
# -----------------------------------------------------------





