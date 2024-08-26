import numpy as np
import matplotlib.pyplot as plt

# y = ax + b
a = 7
b = 2
c = 5

x = np.linspace(-4, 4, 200)
y = a * x + b
plt.plot(x, y, color = "black")

# y = ax^2 + bx + c

y = a * x**2 + b * x + c

plt.plot(x, y, color = "black")

# 데이터 만들기

from scipy.stats import norm, uniform

# 곡선 만들기
k = np.linspace(-4, 4, 200)
sin_y = np.sin(k)

plt.plot(k, sin_y, "red")

# 파란 점들 만들기
np.random.seed(42)
x = uniform.rvs(loc = -4, scale = 8, size = 20)
y = np.sin(x) + norm.rvs(loc = 0, scale = 0.3, size = 20)


plt.plot(k, sin_y, "red")
plt.scatter(x, y, color = "blue")

# train, test 만들기
import pandas as pd

np.random.seed(42)
x = uniform.rvs(loc = -4, scale = 8, size = 30)
y = np.sin(x) + norm.rvs(size = 30, loc = 0, scale = 0.3)

df = pd.DataFrame({
    "x" : x,
    "y" : y
})

train_df = df[:20]
test_df = df[20:]

plt.scatter(train_df["x"], train_df["y"], color = "blue")

# train의 회귀분석
from sklearn.linear_model import LinearRegression

train_x = train_df["x"].values.reshape(-1, 1)
train_y = train_df["y"]

model = LinearRegression()
model.fit(train_x, train_y)

model.coef_
model.intercept_

reg_line = model.predict(train_x)

plt.plot(train_x, reg_line, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")

# 2차 회귀 곡선

train_df["x2"] = train_df["x"] ** 2

x = train_df[["x", "x2"]]
y = train_df["y"]

model = LinearRegression()
model.fit(x, y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200) # 선을 완벽한 곡선으로 그리기 위해

df_k = pd.DataFrame({
    "x" : k, "x2" : k**2
})
df_k

reg_line = model.predict(df_k)

plt.plot(k, reg_line, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")


# 3차 회귀곡선
train_df["x3"] = train_df["x"] ** 3

x = train_df[["x", "x2", "x3"]]
y = train_df["y"]

model = LinearRegression()
model.fit(x, y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200) # 선을 완벽한 곡선으로 그리기 위해

df_k = pd.DataFrame({
    "x" : k, "x2" : k**2, "x3" : k**3
})
df_k

reg_line = model.predict(df_k)

plt.plot(k, reg_line, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")

# 9차 곡선 회귀
train_df["x4"] = train_df["x"] ** 4
train_df["x5"] = train_df["x"] ** 5
train_df["x6"] = train_df["x"] ** 6
train_df["x7"] = train_df["x"] ** 7
train_df["x8"] = train_df["x"] ** 8
train_df["x9"] = train_df["x"] ** 9
train_df

x = train_df[["x", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]]
y = train_df["y"]

model.fit(x, y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x" : k, "x2" : k**2, "x3" : k**3, "x4" : k**4,
    "x5" : k**5, "x6" : k**6, "x7" : k**7, "x8" : k**8, "x9" : k**9
})
df_k
reg_line = model.predict(df_k)

plt.plot(k, reg_line, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")


test_df["x2"] = test_df["x"] ** 2
test_df["x3"] = test_df["x"] ** 3
test_df["x4"] = test_df["x"] ** 4
test_df["x5"] = test_df["x"] ** 5
test_df["x6"] = test_df["x"] ** 6
test_df["x7"] = test_df["x"] ** 7
test_df["x8"] = test_df["x"] ** 8
test_df["x9"] = test_df["x"] ** 9

test_df

y_hat = model.predict(x)

# 9차 모델 성능 : 0.89
sum((test_df["y"] - y_hat) ** 2)

# 20차 선형 회귀 곡선을 그려보자!

