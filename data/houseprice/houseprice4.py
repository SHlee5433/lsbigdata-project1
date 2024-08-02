# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 변수지정 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

house_train["all_round"] = house_train["1stFlrSF"] + (house_train["2ndFlrSF"]) *2/3
house_train["all_round"]

# 필요한 데이터 불러오기기
house_train = pd.read_csv("./data/houseprice/train.csv")
house_test = pd.read_csv("./data/houseprice/test.csv")
sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")

x = np.array(house_train["all_round"]).reshape(-1, 1) # reshape 하기 위해선 꼭 np.array() 실시
y = house_train["SalePrice"] / 1000

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌 

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

house_test["all_round"] = house_test["1stFlrSF"] + (house_test["2ndFlrSF"]) *2/3
house_test["all_round"]


test_x = np.array(house_test["all_round"]).reshape(-1, 1)
test_x


pred_y = model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기 

sub_df["SalePrice"] = pred_y * 1000
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission6.csv", index = False)
sub_df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 그라운드리빙에어리어 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

# 필요한 데이터 불러오기기
house_train = pd.read_csv("./data/houseprice/train.csv")
house_test = pd.read_csv("./data/houseprice/test.csv")
sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")

# 이상치 탐색
house_train = house_train.query("GrLivArea <= 4500")

x = np.array(house_train["GrLivArea"]).reshape(-1, 1) # reshape 하기 위해선 꼭 np.array() 실시
y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌 

# 시각화
# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 5000])
plt.ylim([0, 900000])
plt.legend()
plt.show()
plt.clf()

test_x = np.array(house_test["all_round"]).reshape(-1, 1)
test_x


pred_y = model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기 

sub_df["SalePrice"] = pred_y * 1000
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission6.csv", index = False)
sub_df
