import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# 데이터 로드
house_train = pd.read_csv("data/houseprice/train.csv")
house_test = pd.read_csv("data/houseprice/test.csv")
sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")

# 이상치 제거
house_train = house_train.query('GrLivArea <= 4500')

# 피처와 타겟 변수 선택
x = house_train[['GrLivArea', 'GarageArea']]
y = house_train['SalePrice']

# 모델 학습
model = LinearRegression()
model.fit(x, y)

# 기울기와 절편 출력
slope_grlivarea = model.coef_[0]
slope_garagearea = model.coef_[1]
intercept = model.intercept_

print(f"GrLivArea의 기울기 (slope): {slope_grlivarea}")
print(f"GarageArea의 기울기 (slope): {slope_garagearea}")
print(f"절편 (intercept): {intercept}")

# 3D 그래프 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 데이터 포인트
ax.scatter(x['GrLivArea'], x['GarageArea'], y, color='blue', label='Data points')

# 회귀 평면
GrLivArea_vals = np.linspace(x['GrLivArea'].min(), x['GrLivArea'].max(), 100)
GarageArea_vals = np.linspace(x['GarageArea'].min(), x['GarageArea'].max(), 100)
GrLivArea_vals, GarageArea_vals = np.meshgrid(GrLivArea_vals, GarageArea_vals)
SalePrice_vals = intercept + slope_grlivarea * GrLivArea_vals + slope_garagearea * GarageArea_vals

ax.plot_surface(GrLivArea_vals, GarageArea_vals, SalePrice_vals, color='red', alpha=0.5)

# 축 라벨
ax.set_xlabel('GrLivArea')
ax.set_ylabel('GarageArea')
ax.set_zlabel('SalePrice')

plt.legend()
plt.show()
