# 직선의 방정식
# y+ ax + b
# 예: y = 2x+3 의 그래프를 그려보세요!
a = 1
b = 0

import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-5, 5, 100)
y = a * x + b

plt.plot(x, y, color = "blue")
plt.axvline(0, color = "black") # y축 생성
plt.axhline(0, color = "black") # x축 생성
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()
plt.clf()

import pandas as pd

a = beta_1
b = beta_0

a = 16.38
b = 133.97

x = np.linspace(0, 5, 100)
y = a * x + b


house_train = pd.read_csv("./data/houseprice/train.csv")
my_df = house_train[["BedroomAbvGr", "SalePrice"]].head(10)
my_df["SalePrice"] = my_df["SalePrice"] / 1000
plt.scatter(x = my_df["BedroomAbvGr"], y = my_df["SalePrice"])
plt.plot(x, y, color = "red")
plt.show()
plt.clf()

 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 회귀 계산식 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np

# 예제 데이터
x = my_df["BedroomAbvGr"]
y = my_df["SalePrice"]

# 평균 계산
x_mean = np.mean(x)
y_mean = np.mean(y)

# Sxx 계산
Sxx = np.sum((x - x_mean) * (x - x_mean))

# Sxy 계산
Sxy = np.sum((x - x_mean) * (y - y_mean))

# 결과 출력
print(f"Sxx = {Sxx}")
print(f"Sxy = {Sxy}")

# 회귀 계수 계산
beta_1 = Sxy / Sxx
beta_0 = y_mean - beta_1 * x_mean

beta_1
beta_0
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
a = 16.38
b = 133.97
house_test=pd.read_csv("./data/houseprice/test.csv")
house_test["BedroomAbvGr"]
(a * house_test["BedroomAbvGr"] + b) * 1000


sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df

# SalePrice 바꿔치기 

sub_df["SalePrice"] = (a * house_test["BedroomAbvGr"] + b) * 1000
sub_df

# 파일 내보내기
sub_df.to_csv("./data/houseprice/sample_submission4.csv", index = False)
sub_df

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 직선 성능 평가
a = 47
b = 63

# y_hat 어떻게 구할까? y_hat이란 데이터 1을 회귀직선에 내렸을 때 y좌표
y_hat = (a * house_train["BedroomAbvGr"] + b) * 1000
# y는 어디에 있는가? y는 데이터 1의 y좌표를 말한다.
y = house_train["SalePrice"]

np.abs(y - y_hat)  # 절대거리
np.sum(np.abs(y - y_hat)) # 절대값 합
np.sum((y - y_hat) ** 2) # 제곱합

# 1조 : 106021410
# 2조 : 94512422
# 3조 : 93754868
# 4조 : 81472836
# 5조 : 103493158
# 6조 : 92990794
# 7조 : 
# 회기 (16.38, 133.966)
# !pip install scikit-learn

from sklearn.linear_model import LinearRegression

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

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
plt.legend() # 범례 추가하기기
plt.show()
plt.clf()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 회귀모델을 통한 집값 예측

# 필요한 패키지 불러오기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 필요한 데이터 불러오기기
house_train = pd.read_csv("./data/houseprice/train.csv")
house_test = pd.read_csv("./data/houseprice/test.csv")
sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")

x = np.array(house_train["BedroomAbvGr"]).reshape(-1, 1) # reshape 하기 위해선 꼭 np.array() 실시
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
