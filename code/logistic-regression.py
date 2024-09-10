import pandas as pd
import numpy as np
# 워킹 디렉토리 설정
import os
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)

admission_data = pd.read_csv("../data/admission.csv")
print(admission_data.shape)

# 합격을 한 사건: Admit
# Admit의 확률 오즈(Odds)
# P(Admit) = 합격 인원/ 전체 학생
p_hat = admission_data["admit"].mean()
p_hat / (1 - p_hat)

# P(A): 0.5보다 큰 경우 -> 오즈비 : 무한대에 가까워짐
# P(A): 0.5 -> 오즈비: 1
# P(A): 0.5보다 작은 경우 -> 오즈비: 0에 가까워짐
# 확률의 오즈비가 갖는 값의 범위: 0~무한대

admission_data["rank"].unique()

grouped_data = admission_data \
    .groupby('rank', as_index = False) \
    .agg(p_admit=('admit', 'mean'))

grouped_data['odds'] = grouped_data['p_admit'] / (1 - grouped_data['p_admit'])
print(grouped_data)

# 확룰이 오즈비가 3!
# P(A)

# admission 데이터 산점도 그리기
# x: gre, y: admit
# admission_data

import seaborn as sns

sns.stripplot(data = admission_data, x = "rank", y = "admit", jitter = 0.3, alpha = 0.3)

sns.scatterplot(data = grouped_data, x = "rank", y = "p_admit")

sns.regplot(data = grouped_data, x = "rank", y = "p_admit")

# 로그오즈를 직선회귀로 표현을 하겠다. 그것이 로지스틱회귀분석이다?

odds_data = admission_data.groupby('rank').agg(p_admit=('admit', 'mean')).reset_index()
odds_data['odds'] = odds_data['p_admit'] / (1 - odds_data['p_admit'])
odds_data['log_odds'] = np.log(odds_data['odds'])
print(odds_data)

sns.regplot(data = odds_data, x = "rank", y = "log_odds")

import statsmodels.api as sm
model = sm.formula.ols("log_odds ~ rank", data=odds_data).fit()
print(model.summary())
# log_odds = 종속변수, rank = 독립변수


import statsmodels.api as sm
admission_data['rank'] = admission_data['rank'].astype('category')
admission_data['gender'] = admission_data['gender'].astype('category')
model = sm.formula.logit("admit ~ gre + gpa + rank + gender", data=admission_data).fit()

print(model.summary())

# 입학할 확률의 오즈가 
np.exp(0.7753)

# 여학생
# GPA: 3.5
# GRE: 500
# Rnak: 2
 
# 합격 확률 예측해보세요!
# odds = exp(-3.408 + -0.058 * x1 + 0.002 * x2 + 0.775 * x3 -0.561 * x4)

my_odds = np.exp(-3.408 + -0.058 * 0 + 0.002 * 500 + 0.775 * 3.5 -0.561 * 2)

my_odds / (my_odds+1) # 합격 확률: 0.306
# Odds는?
0.306 / (1 - 0.306) = 0.4409
# 이상태에서 GPA가 1 증가하면 합격 확률 어떻게 변할까?
my_odds = np.exp(-3.408 + -0.058 * 0 + 0.002 * 500 + 0.775 * 4.5 -0.561 * 2)

my_odds / (my_odds+1) # 합격 확률: 0.48937
# Odds는?
0.48937 / (1 - 0.48937) # 0.9583
np.exp(0.7753)

# 여학생
# GPA: 3
# GRE: 450
# Rnak: 2

my_odds = np.exp(-3.408 + -0.058 * 0 + 0.002 * 450 + 0.775 * 3 -0.561 * 2)

my_odds / (my_odds+1) # 합격 확률: 0.2133
# Odds는?
0.2133 / (1 - 0.2133) # 0.271

# 해당 모델의 유의한지 알아보는 방법?
from scipy.stats import norm

2*(1-norm.cdf(2.123, loc=0, scale=1))
2*norm.cdf(-2.123, loc=0, scale=1)

# 우도비 검정의 검정통계량 구하기 (제한된 모형과 비제한 모형 간의 적합도를 비교)
stat_value=-2*(-249.99 - (-229.69))

# p_value 값 구하기
from scipy.stats import chi2

1-chi2.cdf(stat_value, df=4) # df=변수갯수



# ~~~~~~~~~~~~~~~~~~~~~~~~~ 연습 ~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import numpy as np

admission_data = pd.read_csv("../data/admission.csv")
admission_data.shape

# 입학이 허가될 확률의 오즈를 구해보자
p_hat = admission_data["admit"].mean()
p_hat # 입학이 허가될 확률

# 확률의 오즈
p_hat/(1 - p_hat) # 0.465

# 범주형 변수를 사용한 오즈 계산

unique_ranks = admission_data["rank"].unique()
unique_ranks # 1, 2, 3, 4 존재 확인

grouped_data = admission_data \
                    .groupby("rank", as_index = False) \
                    .agg(p_admit = ("admit", "mean"))

grouped_data["odds"] = grouped_data["p_admit"]/(1 - grouped_data["p_admit"])

grouped_data

# 로그 오즈값의 그래프
import numpy as np
import matplotlib.pyplot as plt
p=np.arange(0,1.01,0.01)
log_odds=np.log(p/(1-p))
plt.plot(p, log_odds)

odds_data = admission_data \
                .groupby("rank") \
                .agg(p_admit = ("admit", "mean")).reset_index()

odds_data["odds"] = odds_data["p_admit"] / (1 - odds_data["p_admit"])

odds_data["log_odds"] = np.log(odds_data["odds"])
odds_data


# rank 변수를 수치형이라고 생각하고 회귀직선을 구해보자!
import statsmodels.api as sm

model = sm.formula.ols("log_odds ~ rank", data = odds_data).fit()
model.summary()

# 이러한 직선과 주어진 로그 오즈를 시각화 해보자!

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data = odds_data, x = "rank", y = "log_odds")

sns.regplot(data = odds_data, x = "rank", y = "log_odds", ci = None)

# 로지스틱 회귀분석 해보자!
import statsmodels.api as sm

admission_data["rank"] = admission_data["rank"].astype("category")

admission_data["gender"] = admission_data["gender"].astype("category")

model = sm.formula.logit("admit ~ gre + gpa + rank + gender", data = admission_data).fit()

print(model.summary())
