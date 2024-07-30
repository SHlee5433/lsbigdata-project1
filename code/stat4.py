from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np

# X ~ 균일분포 U(a, b)
# loc : a, scale: b-a
uniform.rvs(loc = 2 scale = 4, size = 1) #loc = 시작점, scale = 구간길이
uniform.pdf(3, loc = 2, scale = 4) 
uniform.pdf(7, loc = 2, scale = 4)

k = np.linspace(0, 8, 100)
y = uniform.pdf(k, loc = 2, scale = 4)
plt.plot(k, y, color = "black")
plt.show()
plt.clf()

# P(X < 3.25) = ?
uniform.cdf(3.25, 2, 4)

# P(5< X < 8.39) = ?
uniform.cdf(8.39, 2, 4) - uniform.cdf(5, 2, 4)

# 상위 7% 값은?
uniform.ppf(0.93, 2, 4)

# 표본 20개 뽑고 표본평균 계산
x = uniform.rvs(loc = 2, scale = 4, size = 20, random_state = 42) 
x. mean()

x = uniform.rvs(loc = 2, scale = 4, size = 20 * 1000, random_state = 42)
x = x.reshape(1000,20)
x.shape
blue_x = x.mean(axis = 1)
blue_x

import seaborn as sns
from scipy.stats import norm
sns.histplot(blue_x, stat = "density")
plt.show()

# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.3333333/20)
uniform.var(loc = 2, scale = 4) # 분산을 구하는 함수
uniform.expect(loc = 2 , scale = 4) # 기댓값 구하는 함수

xmin, xmax = (blue_x.min(), blue_x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.3333333/20))
plt.plot(x_values, pdf_values, color="red", linewidth = 2)

plt.show()
plt.clf()


# 신뢰구간
# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.3333333/20)
from scipy.stats import norm

x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.3333333/20))
plt.plot(x_values, pdf_values, color="red", linewidth = 2)

# 표본평균(파란벽돌) 점찍기
blue_x = uniform.rvs(loc = 2, scale = 4, size =20).mean()
# norm.ppf(0.975, 0, 1)
a = blue_x + 1.96
b = blue_x - 1.96
plt.scatter(blue_x, 0.002,
            color = "blue", zorder = 10, s=10)
plt.axvline(x = a, color = "blue", linestyle = "--", linewidth = 1)
plt.axvline(x = b, color = "blue", linestyle = "--", linewidth = 1)

            
# 기댓값 표현
plt.axvline(x = 4, color = "green",
            linestyle = "-", linewidth = 2)
plt.show()
plt.clf()

# 가운데 면적 95%의 시작점과 끝점인 a, b를 구하시오.
norm.ppf(0.025, 4, np.sqrt(1.3333333/20))
norm.ppf(0.975, 4, np.sqrt(1.3333333/20))

# 가운데 면적 99%의 시작점과 끝점 a, b를 구하시오.
norm.ppf(0.005, 4, np.sqrt(1.3333333/20))
norm.ppf(0.995, 4, np.sqrt(1.3333333/20))








# uniform.pdf(x, loc = 0, scale = 1)
# uniform.cdf(x, loc = 0, scale = 1)
# uniform.ppf(q, loc = 0, scale = 1)
# uniform.rvs(x, loc = 0, scale = 1)
