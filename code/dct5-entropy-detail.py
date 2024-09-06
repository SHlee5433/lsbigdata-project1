import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

# 펭귄 분류 문제
# y: 펭귄의 종류
# x1: bill_length_mm (부리 길이)
# x2: bill_depth_mm (부리 깊이)

df=penguins.dropna()
df=df[["species", "bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={
        "species": "y",
        'bill_length_mm': 'x1',
        'bill_depth_mm': 'x2'})
df

# x1, x2 산점도를 그리되, 점 색깔은 펭귄 종별 다르기 그리기!
import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(data = df, x = "x1", y = "x2", hue = "y")
plt.axvline(x = 45)

# Q. 나누기 전 현재의 엔트로피?
# Q. 45로 나눴을 때, 엔트로피 평균은 얼마인가요?
# 입력값이 벡터 -> 엔트로피!

p_i = df['y'].value_counts()/ len(df['y'])
entropy_curr = -sum(p_i * np.log2(p_i))

# x1 = 45 기준으로 나눈 후, 평균 엔트로피 구하기
# 10분

# x1=45 기준으로 나눈 후, 평균 엔트로피 구하기!
# x1=45 기준으로 나눴을때, 데이터포인트가 몇개 씩 나뉘나요?
n1=df.query("x1 < 45").shape[0]  # 1번 그룹
n2=df.query("x1 >= 45").shape[0] # 2번 그룹

# 1번 그룹은 어떤 종류로 예측하나요?
# 2번 그룹은 어떤 종류로 예측하나요?
y_hat1=df.query("x1 < 45")['y'].mode()
y_hat2=df.query("x1 >= 45")['y'].mode()

# 각 그룹 엔트로피는 얼마 인가요?
p_1=df.query("x1 < 45")['y'].value_counts() / len(df.query("x1 < 45")['y'])
entropy1=-sum(p_1 * np.log2(p_1))

p_2=df.query("x1 >= 45")['y'].value_counts() / len(df.query("x1 >= 45")['y'])
entropy2=-sum(p_2 * np.log2(p_2))

entropy_x145=(n1 * entropy1 + n2 * entropy2)/(n1 + n2)
entropy_x145

# 기준값 x1를 넣으면 entropy값이 나오는 함수는?
# x1 기준으로 최적 기준값은 얼마인가?
def my_entropy(x):
    n1=df.query(f"x1 < {x}").shape[0]  # 1번 그룹
    n2=df.query(f"x1 >= {x}").shape[0] # 2번 그룹   
    p_1=df.query(f"x1 < {x}")['y'].value_counts() / len(df.query(f"x1 < {x}")['y'])
    entropy1=-sum(p_1 * np.log2(p_1))
    p_2=df.query(f"x1 >= {x}")['y'].value_counts() / len(df.query(f"x1 >= {x}")['y'])
    entropy2=-sum(p_2 * np.log2(p_2))
    return float((entropy1 * n1 + entropy2 * n2)/(n1+n2))

my_entropy(45)

result = []
x1_values = np.arange(df["x1"].min(),df["x1"].max()+1,0.01)
for x in x1_values:
    result.append(my_entropy(x))
result
x1_values[np.argmin(result)]

