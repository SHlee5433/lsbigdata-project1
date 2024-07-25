from scipy.stats import bernoulli

# !pip install scipy
# 확률질량함수 pmf
# 확률변수가 갖는 값에 해당하는 확률을 저장하고
# 있는 함수
# bernoulli.pmf(k, p)
# P(X=1)
bernoulli.pmf(1, 0.3)
# P(X=0)
bernoulli.pmf(0, 0.3)

# 이항분포 X ~ P(X= k| n, p)
# n: 베르누이 확률변수 더한 갯수
# p: 1이 나올 확률률
# binom.pmf(k, n, p)
from scipy.stats import binom

binom.pmf(0, n = 2, p = 0.3)
binom.pmf(1, n = 2, p = 0.3)
binom.pmf(2, n = 2, p = 0.3)

# X ~ B(n, p)
# list comp.
result = [binom.pmf(x, n = 30, p = 0.3) for x in range(31)]
result

# numpy
import numpy as np
binom.pmf(np.arange(31), n = 30, p =0.3)

# math
import math
math.factorial(54)/(math.factorial(26) * math.factorial(54 - 26))
math.comb(54, 26)
# ======================== 몰라도 됨 =================================
# 1*2*3*4
# np.cumprod(np.arange(1, 5))[-1]
# fact_54 = np.cumprod(np.arange(1, 55))[-1]
# ln
log(a * b) = log(a) + log(b)
log(1 * 2 * 3 * 4) = log(1) + log(2) + log(3) + log(4)

math.log(24)
sum(np.log(np.arange(1, 5)))

math.log(math.factorial(54))
a = sum(np.log(np.arange(1, 55)))

math.log(math.factorial(26))
b = sum(np.log(np.arange(1, 27)))

math.log(math.factorial(28))
c = sum(np.log(np.arange(1, 29)))

a - (b + c)
np.exp(a - (b + c))
# =================================================================

math.comb(2, 0) * 0.3 ** 0 * (1- 0.3) ** 2
math.comb(2, 1) * 0.3 ** 1 * (1- 0.3) ** 1
math.comb(2, 2) * 0.3 ** 2 * (1- 0.3) ** 0

# pmf : probability mass function (확률질량함수)
binom.pmf(0, 2, 0.3)
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)

# X ~ B(n = 10, p = 0.36)
# P(X = 4)

binom.pmf(4, 10, 0.36)
# P(X <= 4) = ?
sum(binom.pmf(np.arange(5), 10, 0.36))

# P(2< X <=8) = ?
sum(binom.pmf(np.arange(3, 9),10, 0.36))

# X ~ B(n = 30, p = 0.2)
# P(X < 4 or X>=25)

# 1
a = sum(binom.pmf(np.arange(25, 31), 30, 0.2))
# 2
b = sum(binom.pmf(np.arange(4),30, 0.2))
a+b

#4
1 - sum(binom.pmf(np.arange(4, 25), 30, 0.2))

# rvs 함수 (random variates sample)
# 표본 추출 함수
# X1 ~ Bernulli (p=0.3)
bernoulli.rvs(p=0.3)
# X2 ~ Bernulli (p=0.3)
bernoulli.rvs(0.3)
# X ~ B(n=2, p=0.3)
bernoulli.rvs(0.3) + bernoulli.rvs(0.3)
binom.rvs(n=2, p=0.3)
binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)

# X ~ B(30, 0.26)
# 표본 30개를 뽑아보세요!
binom.rvs(n=30, p=0.26, size=30)

# X ~ B(30, 0.26)
# E[X] = ?


# X ~ B(30, 0.26)
# 시각화 해보기!

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
prob_x = binom.pmf(np.arange(31), n =30, p =0.26)
prob_x

x = np.arange(31)
df = pd.DataFrame({"x" : x,
                    "prob" : prob_x})
df

sns.barplot(data = df, x = "x", y = "prob")
plt.show()
plt.clf()


# cdf : cumulative dist. function
# (누적활률분포 함수)
# F_X(x) = P(X <= x)
binom.cdf(4, n=30, p=0.26)
# P(4<x<=18)
binom.cdf(18, n=30, p=0.26) - binom.cdf(4, n=30, p=0.26)
# P(13<x<20)
binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26)


import numpy as np
import seaborn as sns

x_1 = binom.rvs(n=30, p=0.26, size = 10)
x_1
x = np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)
sns.barplot(prob_x, color="blue")

# Add a point at (2, 0)
plt.scatter(x_1, np.repeat(0.002, 10), color = "red", zorder=100, s = 5)

# 기댓값 표현
plt.axvline(x=7.8, color = "green",
            linestyle="--", linewidth=2)
plt.show()
plt.clf()

binom.ppf(0.5, n = 30, p = 0.26)
binom.cdf(8, n = 30, p = 0.26)
binom.cdf(7, n = 30, p = 0.26)
binom.ppf(0.7, n = 30, p = 0.26)
binom.cdf(9, n = 30, p = 0.26)
binom.cdf(8, n = 30, p = 0.26)


1/np.sqrt(2 * math.pi)
from scipy.stats import norm

norm.pdf(0, loc = 0, scale = 1)
norm.pdf(5, loc = 3, scale = 4)

k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc = 3, scale = 1)

plt.plot(k, y, color = "black")
plt.show()
plt.clf()

## mu (loc) : 분포의 중심 결정하는 모수(평균)
k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc = 0, scale = 1)

plt.plot(k, y, color = "black")
plt.show()
plt.clf()

## sigma (scale): 분포의 퍼짐 결정하는 모수(표준편차)
k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc = 0, scale = 1)
y2 = norm.pdf(k, loc = 0, scale = 2)
y3 = norm.pdf(k, loc = 0, scale = 0.5)
plt.plot(k, y, color = "black")
plt.plot(k, y2, color = "red")
plt.plot(k, y3, color = "blue")
plt.show()
plt.clf()


norm.cdf(100, loc = 0, scale = 1)

# P(-2 < X < 0.54) = ?
a = norm.cdf(0.54, loc = 0, scale = 1)
b = norm.cdf(-2, loc = 0, scale = 1)
a-b

# P(X < 1 or X > 3) = ?
x = norm.cdf(1, loc = 0, scale = 1)
y =1 - norm.cdf(3, loc = 0, scale = 1)
x+y

# X ~ N(3,5^2)
# P(3 < X < 5) = 15.54%
norm.cdf(5, 3, 5) - norm.cdf(3, 3, 5)
# 위 확률변수에서 표본 100개 뽑아보자!
x = norm.rvs(loc = 3, scale = 5, size = 1000)
sum((x > 3) & (x < 5))/1000

# 평균: 0, 표준편차: 1
# 표본 1000개 뽑아서 0보다 작은 비율 확인
x = norm.rvs(loc = 0, scale = 1, size = 1000)
sum(x < 0)/1000



x = norm.rvs(loc = 3, scale = 2, size = 1000)
x

sns.histplot(x, stat = "density")


# Plot the normal distribution PDF
xmin, xmax = (x.min(), x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=3, scale=2)
plt.plot(x_values, pdf_values, color="red", linewidth = 2)

plt.show()
plt.clf()
