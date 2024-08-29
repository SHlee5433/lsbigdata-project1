import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자능
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

valid_df = df.loc[20:]
valid_df

for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_x
valid_y = valid_df["y"]
valid_y

from sklearn.linear_model import Lasso

# 결과 받기 위한 벡터 만들기
val_result=np.repeat(0.0, 100) # 0으로 하게 되면 int로 받아서 소수점 표시가 안됨.
tr_result=np.repeat(0.0, 100)


for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x)
    y_hat_val = model.predict(valid_x)

    perf_train = sum((train_df["y"] - y_hat_train)**2)
    perf_val = sum((valid_df["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

tr_result
val_result

import seaborn as sns

df = pd.DataFrame({
    'lambda':np.arange(0, 1, 0.01),
    'tr':tr_result,
    'val':val_result
})

sns.scatterplot(data=df,x='lambda',y='tr') # train set
sns.scatterplot(data=df,x='lambda',y='val', color='red') # valid set
plt.xlim(0,0.4)

val_result[0]
val_result[1]
np.min(val_result)

#alpha를 0.03로 선택!
np.argmin(val_result)
np.arange(0, 1, 0.01)[np.argmin(val_result)]


model = Lasso(alpha = 0.03)
model.fit(train_x, train_y)
model.coef_
model.intercept_

sorted_train=train_x.sort_values("x")
reg_line=model.predict(sorted_train)

plt.plot(sorted_train["x"], reg_line, color="red")
plt.scatter(valid_df["x"], valid_df["y"])


# -4에서 4까지 0.01 간격으로 x 값을 생성합니다.
k=np.linspace(-4, 4, 800)

k_df = pd.DataFrame({
    "x" : k
})

for i in range(2, 21):
    k_df[f"x{i}"] = k_df["x"] ** i
    
k_df

reg_line = model.predict(k_df)

plt.plot(k_df["x"], reg_line, color="red")
plt.scatter(valid_df["x"], valid_df["y"], color="blue")

# 모의고사 5개 보기
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

np.random.seed(42) 
index = np.random.choice(30, 30, replace = False)

a_1 = index[:6]
a_2 = index[6:12]
a_3 = index[12:18]
a_4 = index[18:24]
a_5 = index[24:30]


train_df1 = df.drop(a_1)
train_df2 = df.drop(a_2)
train_df3 = df.drop(a_3)
train_df4 = df.drop(a_4)
train_df5 = df.drop(a_5)


train_df1_x = train_df1[['x']]
train_df2_x = train_df2[['x']]
train_df3_x = train_df3[['x']]
train_df4_x = train_df4[['x']]
train_df5_x = train_df5[['x']]


train_df1_y = train_df1['y']
train_df2_y = train_df2['y']
train_df3_y = train_df3['y']
train_df4_y = train_df4['y']
train_df5_y = train_df5['y']


val_df1 = df.iloc[a_1,:]
val_df2 = df.iloc[a_2,:]
val_df3 = df.iloc[a_3,:]
val_df4 = df.iloc[a_4,:]
val_df5 = df.iloc[a_5,:]


val_df1_x = val_df1[['x']]
val_df2_x = val_df2[['x']]
val_df3_x = val_df3[['x']]
val_df4_x = val_df4[['x']]
val_df5_x = val_df5[['x']]


val_df1_y = val_df1['y']
val_df2_y = val_df2['y']
val_df3_y = val_df3['y']
val_df4_y = val_df4['y']
val_df5_y = val_df5['y']


model = Lasso(alpha = 0.03)
model.fit(train_df1_x, train_df1_y)

model.coef_
model.intercept_

val1_result=np.repeat(0.0, 100)
tr1_result=np.repeat(0.0, 100)

# fold1
for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_df1_x, train_df1_y)

    # 모델 성능
    y_hat_train = model.predict(train_df1_x)
    y_hat_val = model.predict(val_df1_x)

    perf_train=sum((train_df1["y"] - y_hat_train)**2)
    perf_val=sum((val_df1["y"] - y_hat_val)**2)
    tr1_result[i]=perf_train
    val1_result[i]=perf_val

val1_result=np.repeat(0.0, 100)
tr1_result=np.repeat(0.0, 100)

#fold2
val2_result=np.repeat(0.0, 100)
tr2_result=np.repeat(0.0, 100)
for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_df2_x, train_df2_y)

    # 모델 성능
    y_hat_train = model.predict(train_df2_x)
    y_hat_val = model.predict(val_df2_x)

    perf_train=sum((train_df2["y"] - y_hat_train)**2)
    perf_val=sum((val_df2["y"] - y_hat_val)**2)
    tr2_result[i]=perf_train
    val2_result[i]=perf_val

#fold3
val3_result=np.repeat(0.0, 100)
tr3_result=np.repeat(0.0, 100)
for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_df3_x, train_df3_y)

    # 모델 성능
    y_hat_train = model.predict(train_df3_x)
    y_hat_val = model.predict(val_df3_x)

    perf_train=sum((train_df3["y"] - y_hat_train)**2)
    perf_val=sum((val_df3["y"] - y_hat_val)**2)
    tr3_result[i]=perf_train
    val3_result[i]=perf_val

#fold4
val4_result=np.repeat(0.0, 100)
tr4_result=np.repeat(0.0, 100)

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_df4_x, train_df4_y)

    # 모델 성능
    y_hat_train = model.predict(train_df4_x)
    y_hat_val = model.predict(val_df4_x)

    perf_train=sum((train_df4["y"] - y_hat_train)**2)
    perf_val=sum((val_df4["y"] - y_hat_val)**2)
    tr4_result[i]=perf_train
    val4_result[i]=perf_val
#fold 5
val5_result=np.repeat(0.0, 100)
tr5_result=np.repeat(0.0, 100)

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_df5_x, train_df5_y)

    # 모델 성능
    y_hat_train = model.predict(train_df5_x)
    y_hat_val = model.predict(val_df5_x)

    perf_train=sum((train_df1["y"] - y_hat_train)**2)
    perf_val=sum((val_df5["y"] - y_hat_val)**2)
    tr5_result[i]=perf_train
    val5_result[i]=perf_val

tr_result = (tr1_result + tr2_result + tr3_result + tr4_result + tr5_result) / 5
val_result = (val1_result + val2_result + val3_result + val4_result + val5_result) / 5


df = pd.DataFrame({
    'lambda': np.arange(0, 1, 0.01), 
    'tr': tr_result,
    'val': val_result
})

# seaborn을 사용하여 산점도 그리기
sns.scatterplot(data=df, x='lambda', y='tr')
sns.scatterplot(data=df, x='lambda', y='val', color='red')
plt.xlim(0, 0.4)


np.argmin(val_result)
np.arange(0, 1, 0.01)[np.argmin(val_result)]

















