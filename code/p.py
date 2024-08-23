import numpy as np
from matplotlib.pyplot import plt

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

k = np.linspace(-4, 4, 200)

df_k = pd.DataFrame({
    "x" : k, "x2" : k**2
})
df_k

reg_line = model.predict(df_k)

plt.plot(k, reg_line, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")


