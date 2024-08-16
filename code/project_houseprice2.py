# 집값 시각화 코드
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
        
# 데이터 불러오기
file_path = '../lsbigdata-project1/data/houseprice/houseprice-with-lonlat.csv'
house = pd.read_csv(file_path)
house_df = house[["Longitude", "Latitude", "Sale_Price"]]

import folium 
# 흰 도화지 가져오기
house_df["Longitude"].mean()
house_df["Latitude"].mean()

my_map = folium.Map(location = [42.034, -93.642], # 지도 중심 좌표
                    zoom_start = 12, # 확대 단계
                    tiles = "cartodbpositron") # 지도 종류
my_map.save("map_ames.html")

# 점 찍는 법
 price = house_df["Sale_Price"]
  for price, lat, lon in zip(house_df["Sale_Price"],
  house_df["Latitude"],
  house_df["Longitude"]):
      folium.Circle(
         location=[lat, lon],
         radius=50,
         fill=True,
         fill_opacity=0.6,
         popup=f"Price: ${price}",
     ).add_to(my_map) 

my_map.save('map_ames.html')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Sale_Price 정렬
house_sorted = house.sort_values(by='Sale_Price', ascending=False)
house_sorted

hs = house_sorted["Sale_Price"]
hs.head(3)
hs.tail(20)

# 비싼동네 vs 싼동네
house_sorted["Neighborhood"].head(50) #> Northridge, Northridge_Heights, Stone_Brook
house_sorted["Neighborhood"].tail(60) #> Iowa_DOT_and_Rail_Road, Old_Town

# 시각화2(모든 변수)
a = house.groupby('Neighborhood')['Garage_Area'].mean().sort_values()

sns.barplot(a)
plt.title('Average of Garage Area by Neighborhood Type', fontsize=10)
plt.xlabel('Neighborhood', fontsize=7)
plt.ylabel('Average of Garage Area', fontsize=7)

# X축 레이블 각도 및 글씨 크기 조정
plt.xticks(rotation=40, ha='right', fontsize=3)
plt.show()
plt.clf()

# 상위 3개 색 넣는 법
a.tail(4)# 3번 째 값 = 625.7, 4번째 값 = 620
a.head(4)
a = a.reset_index() # 인데스 생성하면서 판다스 프레임으로 변환
type(a)

# house_top3 = oecd1.sort_values("real_wage",ascending=True)
bar_colors = np.where(a["Garage_Area"]>= 625,"red",np.where(a["Garage_Area"] <= 292 ,"blue","grey"))


# plt.figure(figsize=(14, 8))
sns.barplot(data = a, x = "Neighborhood", y = "Garage_Area", \
palette=bar_colors)

plt.title('Average of Garage Area by Neighborhood Type', fontsize=10)
plt.xlabel('Neighborhood', fontsize=7)
plt.ylabel('Average of Garage Area', fontsize=7)

# X축 레이블 각도 및 글씨 크기 조정
plt.xticks(rotation=40, ha='right', fontsize=3)
plt.show()
plt.clf()

# 구 별 위도, 경도 평균에 마커 찍기
# 구 별 위도, 경도 평균 구하기
df = house[["Neighborhood", "Longitude", "Latitude"]]

# groupby로 나누기기
ames = df.groupby("Neighborhood", as_index = False) \
        .agg(mean_lon = ("Longitude", "mean"),
             mean_lat = ("Latitude", "mean")
             )

ames

## 상위 3개 도시 
#> Northridge, Northridge_Heights, Stone_Brook
## 하위 3개 도시시
#> Iowa_DOT_and_Rail_Road, Old_Town
#> # Northridge 구역 표시
# df2 = df[df["Neighborhood"] == "Northridge"].reset_index()
# df2.iloc[35,:]
# 
# folium.Polygon(
#         locations = [[42.054386, -93.651381],[42.048058, -93.650057],[42.047666, -93.653482]],
#         color='#FF8C9E',
#         fill=True,
#         fill_color='#FF8C9E'
#      ).add_to(my_map)
# 
# my_map.save('map_ames.html')

# 흰 도화지 만들기(ames)
my_map = folium.Map(location = [42.034, -93.642], # 지도 중심 좌표
                    zoom_start = 12, # 확대 단계
                    tiles = "cartodbpositron") # 지도 종류

## 원 만들기

folium.Circle(
    location=[42.054386, -93.651381], # 원의 중심 좌표
    radius=100, # 반경 (미터 단위)
    color='orange', # 원 테두리 색상
    fill=True, # 원 내부 색상 채우기
    fill_color='yellow', # 원 내부 색상
    fill_opacity=0.2 # 원 내부 색상의 투명도
).add_to(my_map)

folium.Marker(
    location=[42.054386, -93.651381], # 마커 위치
    popup="expensive", # 팝업 텍스트
    icon=folium.Icon(color='pink', icon='home') # 마커 아이콘 설정
).add_to(my_map)

my_map.save('map_ames.html')



from sklearn.linear_model import LinearRegression

X = house[['Garage_Area']]  # 생활 공간 면적
y = house['Sale_Price']    # 집값

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 예측 값 계산
y_pred = model.predict(X)

# 산점도와 선형 회귀 직선 그리기
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.3, label='Actual Data')  # 실제 데이터 산점도
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')  # 선형 회귀 직선

# 그래프 제목과 축 레이블 설정
plt.title('생활 공간 면적과 집값 간의 관계 (선형 회귀)', fontsize=16)
plt.xlabel('생활 공간 면적 (GrLivArea)', fontsize=14)
plt.ylabel('집값 (SalePrice)', fontsize=14)
plt.legend()
plt.show()
plt.clf()

# ~~~~~~~~~~~~~~~~~~~~ 대쉬보드

model.coef_
model.intercept_

fig.add_trace(
    go.Scatter(
    mode = "lines",
    x = house['Garage_Area'], y = y_pred,
    name = "선형회귀직선",
    line = dict(dash = "dot", color = "red")
    )
)

fig.show()
