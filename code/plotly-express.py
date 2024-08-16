# 데이터 패키지 설치
# !pip install palmerpenguins
import pandas as pd
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins




penguins = load_penguins()
penguins.head()
# x : bill_length_mm
# y : bill_depth_mm
# margins_P = {"l": 25, "r": 25, "t": 50, "b": 50}
fig = px.scatter(penguins, x = "bill_length_mm", y = "bill_depth_mm",
                 color = "species", # trendline = "ols", # p.134
                 ) # p.134

# 1. 제목 크기 키울것,
# 2. 점 크기 크게,
# 3. 범례 제목 "펭귄 종"으로 변경
## layout 설정
fig.update_layout(
    title=dict(text="팔머펭귄 종별 부리 길이 vs. 깊이", font=dict(color="white", size = 24)),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
     legend=dict(
        title=dict(text="펭귄 종"),  # 범례 제목 설정
        font=dict(color="white")),
)
fig.update_traces(marker=dict(size=10, opacity = 0.5))  # 점 크기, 투명도 설정

fig.show()


from sklearn.linear_model import LinearRegression

model = LinearRegression()
penguins = penguins.dropna()
x = penguins[["bill_length_mm"]]
y = penguins["bill_depth_mm"]

model.fit(x, y)
linear_fit = model.predict(x) # 빨간색 선을 그리기 위해 x에 대응하는 y값

model.coef_
model.intercept_

fig.add_trace(
    go.Scatter(
    mode = "lines",
    x = penguins["bill_length_mm"], y = linear_fit,
    name = "선형회귀직선",
    line = dict(dash = "dot", color = "white")
    )
)

fig.show()

# 범주형 변수로 회귀분석 진행하기
# 범주형 변수인 'species'를 더미 변수로 변환
penguins_dummies = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=True)
penguins_dummies.columns
penguins_dummies.iloc[:,-3:] # 뒤 칼럼 3개를 다 가져와라


# x와 y 설정
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
y = penguins_dummies["bill_depth_mm"]

# 모델 학습
model = LinearRegression()
model.fit(x, y)

model.coef_
model.intercept_

regline_y = model.predict(x)

import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(x = x["bill_length_mm"], y = y, size = 3,
                hue = penguins["species"], palette = "deep",
                legend = False)
sns.scatterplot(x = x["bill_length_mm"], y = regline_y,
                color = "black")
plt.show()
plt.clf()
    
# y = 0.2 * bill_length - 1.93 * species_Chinstrap -5.1 * species_Gentoo + 10.56


