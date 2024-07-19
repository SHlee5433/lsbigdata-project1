import numpy as np
import matplotlib.pyplot as plt

# 예제 넘파이 배열 생성
data = np.random.rand(5)

# 히스토그램 그리기
plt.clf()
plt.hist(data, bins = 30, alpha = 0.7, color = "blue")
plt.title("histogram of Numpy Vector")
plt.xlabel("value")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()



x = np.random.rand(10000, 5).mean(axis =1)
x = np.random.rand(50000) \
      .reshape(-1, 5) \
      .mean(axis = 1)
x
