from scipy.stats import binom


# Y ~ B(3, 0.7)
# Y가 갖는 값에 대응하는 확률은?
# P(Y = 0)
binom.pmf(0, 3, 0.7)

import numpy as np
binom.pmf(np.array([0, 1, 2, 3]), 3, 0.7)

# Y ~ B(20, 0.45)
# P(6 < Y <= 14) = ?

sum(binom.pmf(np.arange(7, 15), 20, 0.45))
binom.cdf(14, 20, 0.45) -  binom.cdf(6, 20, 0.45)

# X ~ N(30, 4^2)
# P(X > 24) = ?
from scipy.stats import norm
1 - norm.cdf(24, loc = 30, scale = 4)

# 표본은 8개를 뽑아서 표본평균 X bar
# P(28 < X_bar < 29.7) =?

a = norm.cdf(29.7, loc = 30, scale = 4/np.sqrt(8))
b = norm.cdf(28, loc = 30, scale = 4/np.sqrt(8))

a-b

# 자유도 7인 카이제곱분포 확률밀도 함수 그리기
from scipy.stats import chi2
import matplotlib.pyplot as plt
k = np.linspace(-2, 20, 100)
y = chi2.pdf(k, 7)
plt.plot(k, y, color = "black")


