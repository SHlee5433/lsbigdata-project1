import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 데이터 로드
house_train = pd.read_csv("data/houseprice/train.csv")
house_test = pd.read_csv("data/houseprice/test.csv")
sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")

# 이상치 탐색
# house_train = house_train.query('GrLivArea <= 4500')


#숫자형 변수만 선택하기
x = house_train.select_dtypes(include=[int,float])

# 필요없는 칼럼 제거
x = x.iloc[:, 1:-1]

# x = house_train[['GrLivArea', 'GarageArea']]
y = house_train['SalePrice']
x.isna().sum()

# 결측치 제거 후 채우기
x["LotFrontage"] = x["LotFrontage"].fillna(house_train["LotFrontage"].mean())
x["MasVnrArea"] = x["MasVnrArea"].fillna(house_train["MasVnrArea"].mean())
x["GarageYrBlt"] = x["GarageYrBlt"].fillna(house_train["GarageYrBlt"].mean())

x.isna().sum()

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

test_x = house_test.select_dtypes(include=[int,float])
test_x = test_x.iloc[:, 1:]
test_x.isna().sum()

# fill_values = {
#     "LotFrontage" : test_x["LotFrontage"].mean(),
#     "MasVnrArea" : test_x["MasVnrArea"].mean(),
#     "GarageYrBlt" : test_x["GarageYrBlt"].mean()
# }

# 결측치 채우기
# test_x = test_x.fillna(value = fill_values)
test_x = test_x.fillna(test_x.mean())


# 결측치 확인
test_x.isna().sum()

# 테스트 데이터 집값 예측
pred_y = model.predict(test_x) # test 셋에 대한 집값
pred_y


# SalePrice 바꿔치기 
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission8.csv", index = False)
sub_df

