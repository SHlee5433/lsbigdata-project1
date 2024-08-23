import numpy as np
import matplotlib.pyplot as plt

# y = ax^2 + bx + c 그래프 그리기
a = 7
b = 2
c = 5

x = np.linspace(-8, 8, 100)
y = a * x ** 2 + b * x + c
plt.plot(x, y, color = "black")
plt.show()
plt.clf()

# y = ax^3 + bx^2 + cx + d 그래프
a = 1
b = 0
c = -10
d = 0
e = 10
x = np.linspace(-4, 4, 1000)
# y = a * x ** 3 + b * x ** 2 + c * x + d
y = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e
plt.plot(x, y, color = "black")
plt.show()
plt.clf()

# 데이터 만들기
from scipy.stats import norm
from scipy.stats import uniform


# 검정 곡선
k = np.linspace(-4, 4, 20)
sin_y = np.sin(k)

# 파란 점들
x = uniform.rvs(size = 20, loc = -4, scale = 8)
y = np.sin(x) + norm.rvs(size = 20, loc = 0, scale = 0.3)

plt.plot(k, sin_y, color = "black")
plt.scatter(x, y, color = "blue")

# train, test 데이터 만들기
np.random.seed(42)
x = uniform.rvs(size = 30, loc = -4, scale = 8)
y = np.sin(x) + norm.rvs(size = 30, loc = 0, scale = 0.3)

import pandas as pd
df = pd.DataFrame({
    "x" : x, "y" : y
})
df
train_df = df.loc[:19]
train_df

test_df = df.loc[20:]
test_df

plt.scatter(train_df["x"], train_df["y"], color = "blue")

# train의 회귀분석
from sklearn.linear_model import LinearRegression
x_train = train_df["x"].values.reshape(-1, 1) # train_df[["x"]]도 가능
y_train = train_df["y"]

model = LinearRegression()
model.fit(x_train, y_train)

model.coef_
model.intercept_

reg_line = model.predict(x_train)
plt.plot(x_train, reg_line, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")

# 2차 곡선 회귀
train_df["x2"] = train_df["x"] ** 2
train_df

x = train_df[["x", "x2"]]
y = train_df["y"]

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

# 3차 곡선 회귀

train_df["x3"] = train_df["x"] ** 3
train_df

x = train_df[["x", "x2", "x3"]]
y = train_df["y"]

model.fit(x, y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x" : k, "x2" : k**2, "x3" : k**3
})
df_k
reg_line = model.predict(df_k)

plt.plot(k, reg_line, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")

# 4차 곡선 회귀

train_df["x4"] = train_df["x"] ** 4
train_df

x = train_df[["x", "x2", "x3", "x4"]]
y = train_df["y"]

model.fit(x, y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x" : k, "x2" : k**2, "x3" : k**3, "x4" : k**4
})
df_k
reg_line = model.predict(df_k)

plt.plot(k, reg_line, color = "red")
plt.scatter(train_df["x"], train_df["y"], color = "blue")

# 9차 곡선 회귀

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
