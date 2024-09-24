# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

cwd = os.getcwd() 
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

## 이상치 탐색 (여기 넣으면 안됨)
# house_train = house_train.query("GrLivArea <= 4500")

house_train.shape
house_test.shape
train_n = len(house_train)

# 통합 df 만들기 
df = pd.concat([house_train, house_test], ignore_index = True)
df

df = pd.get_dummies(
    df,
    columns = ["Neighborhood"],
    drop_first = True
)
df

# train / test 데이터셋
train_df = df.iloc[:train_n,]
test_df = df.iloc[train_n:,]

# Validation 셋(모의고사 셋) 만들기
# np.random.randint(0, 1459, size = 438) 중복된 값이 뽑힐 수 도 있음
np.random.seed(42)
val_index = np.random.choice(np.arange(train_n), size = 438, replace = False)
val_index

# train => valid / train 데이터 셋
valid_df = train_df.loc[val_index] # 30%
train_df = train_df.drop(val_index)# 70%

## 이상치 탐색
train_df = train_df.query("GrLivArea <= 4500")

# x, y 나누기
# ^ 시작을 의미, $ 끝남을 의미, | or를 의미
selected_columns = train_df.filter(regex = "^GrLivArea$|^GarageArea$|^Neighborhood").columns
train_x = train_df[selected_columns]
train_y = train_df["SalePrice"]

valid_x = valid_df[selected_columns]
valid_y = valid_df["SalePrice"]
# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y) # 자동으로 기울기, 절편 값을 구해줌

# 성능 측정
y_hat = model.predict(valid_x)
np.sqrt(np.mean((valid_y - y_hat)**2))




## 이상치 탐색
# house_train=house_train.query("GrLivArea <= 4500")

## 회귀분석 적합(fit)하기
# house_train["GrLivArea"]   # 판다스 시리즈
# house_train[["GrLivArea"]] # 판다스 프레임
house_train["Neighborhood"]
neighborhood_dummies = pd.get_dummies(
    house_train["Neighborhood"],
    drop_first=True
    )
# pd.concat([df_a, df_b], axis=1)
x= pd.concat([house_train[["GrLivArea", "GarageArea"]], 
             neighborhood_dummies], axis=1)
y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

neighborhood_dummies_test = pd.get_dummies(
    house_test["Neighborhood"],
    drop_first=True
    )
test_x= pd.concat([house_test[["GrLivArea", "GarageArea"]], 
                   neighborhood_dummies_test], axis=1)
test_x

# 결측치 확인
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)