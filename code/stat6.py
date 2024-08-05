# y = 2matplotlib.pyplot# y = 2x + 3 그리기
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
# x값의 범위 설정
x = np.linspace(0, 100, 400)

# y 값 계산
y = 2 * x + 3

# np.random.seed(20240805)
obs_x = np.random.choice(np.arange(100), 20)
epsilon_i = norm.rvs(loc = 0, scale = 300, size = 20)
obs_y = 2 * obs_x + 3 + epsilon_i

# 그래프 그리기
plt.plot(x, y, label="y = 2x + 3", color = "black")
plt.scatter(obs_x, obs_y, color = "blue", s = 3)
# plt.show()
# plt.clf()

from sklearn.linear_model import LinearRegression


# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
obs_x = obs_x.reshape(-1, 1)
model.fit(obs_x, obs_y)

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a hat
model.intercept_ # 절편 b hat

# 회귀 직선 그리기
x = np.linspace(0, 100, 400)
y = model.coef_ * x + model.intercept_
plt.xlim([0, 100])
plt.ylim([0, 300])
plt.plot(x, y, color = "red") # 회귀직선선
plt.show()
plt.clf()

!pip install statsmodels
import statsmodels.api as sm

obs_x = sm.add_constant(obs_x)
model = sm.OLS(obs_y, obs_x).fit()    
print(model.summary())


1 - norm.cdf(18, loc = 10, scale = 1.96)
1 - norm.cdf(4.08, loc = 0, scale = 1)

