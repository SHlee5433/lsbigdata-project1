import numpy as np

mat_a = np.array([14, 4, 0, 10]).reshape(2, 2)
mat_a

o_ij = np.array([14, 4, 0, 10])
# 귀무가설: 두 변수 독립
# 대립가설: 두 변수가 독립 x
from scipy.stats import chi2_contingency

chi2, p, df, expected = chi2_contingency(mat_a, correction=False)
chi2.round(3) # 검정통계량
p.round(4) # p - value

np.sum((mat_a - expected) **2 / expected)

# 유의수준 0.05 이라면,
# p 값이 0.05보다 작으므로, 귀무가설을 기각
# 즉, 두 변수는 독립 아니다.
# X ~ chi2(1)일 때, P(X > 15.556) = ?
from scipy.stats import chi2

1- chi2.cdf(15.556, df = 1)
p

# 귀무가설: 두 도시에서의 음료 선호도가 동일하다.
# 대립가설: 두 도시에서의 음료 선호도가 동일하지 않다.
mat_b = np.array([[50, 30, 20],
                  45, 35, 20])
mat_b

chi2, p, df, expected = chi2_contingency(mat_b, correction = False)
chi2.round(3) # 검정 통계량
p.round(4)    # p-value
expected

# 귀무가설: 정당 지지와 핸드폰 사용 유무는 독립이다.
# 대립가설: 정당 지지와 핸드폰 사용 유무는 독립이 아니다.

mat_c = np.array([49, 47, 15, 27, 32, 30]).reshape(3, 2)
mat_c

chi2, p, df, expected = chi2_contingency(mat_c, correction= False)
chi2.round(3) # 검정 통계량
p.round(4)    # p-value
expected

from scipy.stats import chisquare
import numpy as np

observed = np.array([13, 23, 24, 20, 27, 18, 15])
expected = np.repeat(20, 7)
statistic, p_value = chisquare(observed, f_exp = expected)

statistic
# 7.6

# 자유도 6일 때, 7.6를 넘을 확률 = 0.268
p_value


# 지역별 후보 지지율
# 귀무가설: 선거구별 지지율이 동일하다
# 대립가설: 선거구별 지지율이 동일하지 않은게 1개라도 있다.
mat_b = np.array([[176, 124],
                  [193, 107],
                  [159, 141]])
mat_b

chi2, p, df, expected = chi2_contingency(mat_b, correction = False)
chi2.round(3) # 검정 통계량
p.round(4)    # p-value
expected

