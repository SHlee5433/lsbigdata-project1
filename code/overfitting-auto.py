import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자능
np.random.seed(42)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "x" : x , "y" : y
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
y = train_df["y"]

model=LinearRegression()
model.fit(x,y)

test_df = df.loc[20:]
test_df

for i in range(2, 21):
    test_df[f"x{i}"] = test_df["x"] ** i
    
# f 스트링 사용법
#i = 3.141592
#f"x{i:.2f}"

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
x = test_df[["x"] + [f"x{i}" for i in range(2, 21)]]

y_hat = model.predict(x)

# 모델 성능
sum((test_df["y"] - y_hat)**2)