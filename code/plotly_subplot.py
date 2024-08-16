import pandas as pd
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()
# penguins.head()

fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    # trendline="ols" # p.134 # 선형회귀 직선 만들기
)
fig.show()

# subplot 관련 패키지 가져오기

from plotly.subplots import make_subplots

fig_subplot = make_subplots(
    rows = 1, cols = 3,
    subplot_titles = ("Adeli", "Gentoo", "Chinstrap")
)

fig_subplot.add_trace(
    {
        "type" : "scatter",
        "mode" : "markers",
        "x" : penguins.query('species == "Adelie"')["bill_length_mm"],
        "y" : penguins.query('species == "Adelie"')["bill_depth_mm"],
        "name": "Adelie"
    },
    row = 1, col = 1
)

fig_subplot.add_trace(
    {
        "type" : "scatter",
        "mode" : "markers",
        "x" : penguins.query('species == "Gentoo"')["bill_length_mm"],
        "y" : penguins.query('species == "Gentoo"')["bill_depth_mm"],
        "name": "Gentoo"
    },
    row = 1, col = 2
)

fig_subplot.add_trace(
    {
        "type" : "scatter",
        "mode" : "markers",
        "x" : penguins.query('species == "Chinstrap"')["bill_length_mm"],
        "y" : penguins.query('species == "Chinstrap"')["bill_depth_mm"],
        "name": "Chinstrap"
    },
    row = 1, col = 3
)

fig_subplot.update_layout(
    title = dict(text = "펭귄 종 별 부리길이 vs 깊이",
    x = 0.5)
)


# penguins["bill_length_mm"].query() 가 안되는 이유 = query는 데이터 프레임만 지원해주기 때문 위 문장은 시리즈!


# 세팅에 들어가는 단축키 : Ctrl + ,
# 함수 도움말 보는 법 : 함수이름 뒤에 ? 붙여서 실행

species_list = ["Adelie", "Gentoo", "Chinstrap"]
for i, species in enumerate(species_list, start = 1) :
    print(i)
    print(species)

# Using the penguins dataset for plotting
# Create a scatter plot using plotly.subplots with Korean annotations, centering titles, and sharing x-axis across subplots
# All x and y axis tick labels are displayed

# Using the penguins dataset for plotting
# Create a scatter plot using plotly.subplots with Korean annotations, centering titles, and sharing x-axis across subplots
# All x and y axis tick labels are displayed, with consistent x-axis range across subplots

fig = make_subplots(rows=1, cols=3, subplot_titles=["아델리", "친스트랩", "젠투"], horizontal_spacing=0.05, shared_xaxes=True)

# Determine the overall range of bill lengths to set a consistent x-axis range
min_bill_length = penguins['bill_length_mm'].min()
max_bill_length = penguins['bill_length_mm'].max()

# Loop through the unique species to create plots
for i, species in enumerate(penguins['species'].unique(), 1):
    subset = penguins[penguins['species'] == species]
    fig.add_trace(
        go.Scatter(
            x=subset['bill_length_mm'], 
            y=subset['bill_depth_mm'], 
            mode='markers',
            marker=dict(size=7, line=dict(width=1)),
            name=f'{species}'
        ),
        row=1, col=i
    )

# Update xaxis and yaxis properties for all subplots
fig.update_xaxes(title_text="부리 길이 (mm)", range=[min_bill_length, max_bill_length], tickmode='auto')
fig.update_yaxes(title_text="부리 깊이 (mm)", tickmode='auto')

# Update layout and size, center title
fig.update_layout(height=400, width=1000, title_text="펭귄 종별 부리 치수", title_x=0.5)
fig.update_layout(showlegend=False)  # Hide legend for clarity

# Enable all x and y axes to show tick labels
fig.update_xaxes(showticklabels=True)
fig.update_yaxes(showticklabels=True)

fig


# 레이아웃 배치 조절
# Using the penguins dataset for plotting
# Create a scatter plot using plotly.subplots with a large total plot and three species-specific subplots below
# The large plot occupies 2 rows and the species-specific plots occupy 1 row

fig = make_subplots(
    rows=3, cols=3,
    specs=[[{'colspan': 3, 'rowspan': 2}, None, None],
           [None, None, None],
           [{'colspan': 1}, {'colspan': 1}, {'colspan': 1}]],
           # This row is needed to accommodate the rowspan of 2 for the total plot
    subplot_titles=["전체 데이터", "아델리", "친스트랩", "젠투"],
    row_heights=[0.4, 0.4, 0.2],  # Heights for the total plot and the species-specific plots
    shared_xaxes=True,
    horizontal_spacing=0.05, # plot과 plot 차이 공간을 얼마나 줄 것인가
    vertical_spacing=0.05) # x축과 점 사이의 공간을 얼마나 줄 것인가

# Colors for different species
colors = {
    "Adelie": "blue",
    "Chinstrap": "red",
    "Gentoo": "green"
}

# Plot for all data with different colors for each species
import plotly.graph_objects as go
for species, color in colors.items():
    subset = penguins[penguins['species'] == species]
    fig.add_trace(
        go.Scatter(
            x=subset['bill_length_mm'], 
            y=subset['bill_depth_mm'], 
            mode='markers',
            marker=dict(size=8, color=color),
            name=f'{species}'
        ),
        row=1, col=1
    )

# Subplots for each species in the third logical row
for i, species in enumerate(penguins['species'].unique(), 1):
    subset = penguins[penguins['species'] == species]
    fig.add_trace(
        go.Scatter(
            x=subset['bill_length_mm'], 
            y=subset['bill_depth_mm'], 
            mode='markers',
            marker=dict(size=7, line=dict(width=1), color=colors[species]),
            name=f'{species}'
        ),
        row=3, col=i
    )

# Update xaxis and yaxis properties for all subplots
fig.update_xaxes(title_text="부리 길이 (mm)")
fig.update_yaxes(title_text="부리 깊이 (mm)")

# Update layout and size, center title
fig.update_layout(height=900, width=1000, title_text="펭귄 종별 부리 치수", title_x=0.5)
fig