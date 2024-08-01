# X~ N(3, 7^2)

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

x = norm.ppf(0.25, loc = 3, scale = 7)
z = norm.ppf(0.25, loc = 0, scale = 1)

x
3 + z * 7

norm.cdf(5, loc = 3, scale = 7)
norm.cdf(2/7, loc = 0, scale = 1)


# 표본 1000개, 히스토그램 -> pdf 겹쳐서 그리기 X ~ N(0, 1)
z = norm.rvs(loc = 0, scale = 1, size = 1000)
z

sns.histplot(z, stat = "density", color = "grey")


# Plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color="red", linewidth = 2)

plt.show()

# X ~ N(3,sqrt(2)^2)
z_2 = np.sqrt(2) * z + 3

sns.histplot(z_2, stat = "density", color = "green")


# Plot the normal distribution PDF
z_2min, z_2max = (z_2.min(), z_2.max())
z_2_values = np.linspace(z_2min, z_2max, 100)
pdf_values = norm.pdf(z_2_values, loc=3, scale=np.sqrt(2))
plt.plot(z_2_values, pdf_values, color="blue", linewidth = 2)

plt.show()
plt.clf()


# 표준화 확인
# X ~ N(5, 3^2)

x = norm.rvs(loc = 5, scale = 3, size = 1000)
x
# 표준화
z = (x - 5) / 3
sns.histplot(z, stat = "density", color = "grey")


# Plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color="red", linewidth = 2)

plt.show()
plt.clf()

# X ~ N(5, 3^2)
# z = (x - 5) / 3 가 표준정규분포를 따르나요?
# 표본표준편차로 나눠도 표준정규분포가 될까?
# 1.
x = norm.rvs(loc = 5, scale = 3, size = 10) # 표본이 너무 적어 불안정함함
x
# 표본 분산값 구하기
s = np.std(x, ddof = 1)
s_2 = s ** 2

# 2.
x = norm.rvs(loc = 5, scale = 3, size = 1000)

# 3. 표준화 하기
z = (x - 5) / s
z

sns.histplot(z, stat = "density", color = "grey")

zmin,zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc = 0, scale = 1)
plt.plot(z_values, pdf_values, color = "r", linewidth = 3)
plt.show()
plt.clf()

# t 분포에 대해서 알아보자!
X ~ t(df)
# 종모양, 대칭분포, 중심 0
# 모수 df : 자유도라고 부름 - 퍼짐을 나타내는 모수
# df 이 작으면 분산 커짐
# df 이 무한대로 가면 표준정규분포
from scipy.stats import t

# t.pdf
# t.cdf
# t.ppf
# t.rvs
# 자유도가 4인 t분포의 pdf를 그려보세요!

t_values = np.linspace(-4, 4, 100)
pdf_values = t.pdf(t_values, df = 30)
plt.plot(t_values, pdf_values, color="red", linewidth = 2)

# 표준정규분포 그리기
pdf_values = norm.pdf(t_values, loc=0, scale=1)
plt.plot(t_values, pdf_values, color="black", linewidth = 2)


plt.show()
plt.clf()

# X ~ ?(mu, sigma^2)
# X bar ~ N(mu, sigma^2/n)
# X_bar ~= N(x_bar, s^2/n) 자유도가 n-1인 t 분포
x = norm.rvs(loc = 15, scale = 3, size = 16, random_state = 42)
x
x_bar = x.mean()
n = len(x)
# df = 자유도(degree of freedom)
# 모분산을 모를때 : 모평균에 대한 95% 신뢰구간을 구해보자!
x_bar + t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)
x_bar - t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)

# 모분산(3^2)을 알때 : 모평균에 대한 95% 신뢰구간을 구해보자!
x_bar + norm.ppf(0.975, loc = 0, scale = 1) * 3 / np.sqrt(n)
x_bar - norm.ppf(0.975, loc = 0, scale = 1) * 3 / np.sqrt(n)



