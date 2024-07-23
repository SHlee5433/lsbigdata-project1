import numpy as np
import matplotlib.pyplot as plt

# 예제 넘파이 배열 생성
data = np.random.rand(5)

# 히스토그램 그리기
plt.clf()
x = np.random.rand(10000, 5).mean(axis =1)
plt.hist(data, bins = 30, alpha = 0.7, color = "blue")
plt.title("histogram of Numpy Vector")
plt.xlabel("value")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()


import numpy as np

x = np.arange(33)
sum(x)/33
sum((x - 16) * 1/33)
(x - 16) ** 2

np.unique((x - 16)**2) * (2/33)
sum(np.unique((x - 16)**2) * (2/33))

# E[X^2]
sum(x**2 * (1/33))

# Var(X) = E[X^2] - (E[X])^2
sum(x**2 * (1/33)) - 16 **2

## Example
x = np.arange(4)
x
pro_x = np.array([1/6, 2/6, 2/6, 1/6])
sum(x * pro_x)

# 기댓값
Ex = sum(x * pro_x)
Exx = sum(x**2 * pro_x)

# 분산
Exx - Ex ** 2

sum((x - Ex) * 2 * pro_x)

## Example2
x = np.arange(99)
x
# 1-50-1 벡터
x_1_50_1 = np.concatenate((np.arange(0, 51), np.arange(49, 0, -1)))
pro_x = x_1_50_1/2500
pro_x

## Example3 X: 0, 2, 4, 6
x = np.arange(0, 4)*2
pro_x = np.array([1/6, 2/6, 2/6, 1/6])
pro_x

# 기댓값
Ex = sum(x * pro_x)
Exx = sum(x**2 * pro_x)

# 분산
Exx - Ex ** 2

sum((x - Ex) ** 2 * pro_x)

9.52 ** 2/16

np.sqrt(9.52 ** 2/16)
np.sqrt(9.52 ** 2/10)
