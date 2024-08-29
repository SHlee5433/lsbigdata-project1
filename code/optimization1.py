# y = (x-2) ^2 + 1 그래프 그리기
# 점을 직선으로 이어서 표현
import matplotlib.pyplot as plt
import numpy as np

k =2
x = np.linspace(-4, 8, 100)
y = (x - 2) ** 2 + 1
# plt. scatter(x, y, s = 3)
line_y = 4 * x - 11
plt.plot(x, y, color = "blue")
plt.xlim(-4, 8)
plt.ylim(0, 15)

# x가 주어지면 접선이 알아서 만들어지게 해보자!

# f'(x) = 2x - 4
# k = 4의 기울기
l_slope = 2*k - 4                 # 기울기
f_k = (k - 2) ** 2 + 1            # f'(x)
l_intercept = f_k - l_slope * k   # y절편

# y = slope * x + intercept
line_y = l_slope * x + l_intercept
plt.plot(x, line_y, color = "red")

# 경사하강법
# y = x^2, 초기값 10, 델타 : 0.9일때, x(100) = 100번째

x = 10
lstep = np.arange(100, 0, -1) * 0.01
for i in range(100) :
    x = x-lstep[i] * (2 * x)

x

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(x, y, z, levels=20)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# 특정 점 (9, 2)에 빨간색 점을 표시
plt.scatter(9, 2, color = "red", s =50)

x = 9; y = 2
lstep = 0.1
for _ in range(100) : 
    (x, y) = np.array([x, y]) - lstep * np.array([2*x -6, 2*y-8])
    plt.scatter(float(x), float(y), color = "red", s = 25)
print(x,y)


# 축 레이블 및 타이틀 설정
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of f(x, y) = (x-3)^2 + (y-4)^2 + 3')
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# 그래프 표시
plt.show()

# x를 기준으로 미분 8x + 20y -23
# y를 기준으로 미분 60y+20x−67

# Q. 다음을 최소로 만드는 베타 벡터
# f(beta0, beta1) = (1-(beta0+beta1))^2 +
#                   (4-(beta0+2*beta1))^2 +
#                   (1.5-(beta0+3*beta1))^2 +
#                   (5-(beta0+4*beta1))^2
# 초기값 : (10,10)
# delta : 0.01

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# beta0, beta1의 값을 정의합니다 
beta0 = np.linspace(-20, 20, 100)
beta1 = np.linspace(-20, 20, 100)
beta0, beta1 = np.meshgrid(beta0, beta1)

# 함수 f(beta0, beta1)를 계산합니다.
z = (1-(beta0+beta1))**2 + (4-(beta0+2*beta1))**2 + (1.5-(beta0+3*beta1))**2 + (5-(beta0+4*beta1))**2

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(beta0, beta1, z, levels=100)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# 특정 점 (10,10)에 빨간색 점을 표시
plt.scatter(9, 9, color="red", s=50)

# x(100), y(100) 구하는 방법
beta0 = 9; beta1 = 9
lstep = 0.01
for i in range(1000) : 
    (beta0, beta1) = np.array([beta0, beta1]) - lstep * np.array([8*beta0 + 20*beta1 -23, 20*beta0 + 60*beta1 -67])
    plt.scatter(beta0,beta1,color="red", s=50)

print(beta0,beta1)

#축 레이블 및 타이틀 설정
plt.xlabel("beta0")
plt.ylabel("beta1")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
# 그래프 표시
plt.show()

# 모델 fit으로 베타 구하기
import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.DataFrame({
    "x" : np.array([1, 2, 3, 4]),
    "y" : np.array([1, 4, 1.5, 5])
})

model = LinearRegression()
model.fit(df[["x"]], df["y"])

model.coef_
model.intercept_

# 날개길이 15, 부리깊이 5.6인 펭귄 부리길이는?


