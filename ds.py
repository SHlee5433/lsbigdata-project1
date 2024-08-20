# !pip install folium
import pandas as pd
import folium
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import os
# # current working directory
cwd = os.getcwd()
cwd
# Ames 데이터 불러오기
df = pd.read_csv("lsbigdata-project1/data/Ames_Project/houseprice/houseprice-with-lonlat.csv")

# ames 중심 위도, 경도 구하기
df['Longitude'].mean()
df['Latitude'].mean()

# 흰 도화지 맵 그리기
ames_map = folium.Map(location = [42.034, -93.642],
                    zoom_start = 12,
                    tiles='cartodbpositron')
ames_map.save('../maps/ames_map.html')
================================================================
import folium
import branca.colormap as cm
import matplotlib.pyplot as plt

# 지도 생성
ames_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12, tiles='cartodbpositron')

# 집값의 최소값과 최대값 구하기
min_price = df['Sale_Price'].min()
max_price = df['Sale_Price'].max()

# 색상맵 정의 (파란색 -> 빨간색)
colormap = cm.LinearColormap(colors=['blue', 'red'], vmin=min_price, vmax=max_price)
colormap.caption = 'House Price'

for _, row in df.iterrows():
    # 집값을 0에서 1 사이의 값으로 정규화
    normalized_price = (row['Sale_Price'] - min_price) / (max_price - min_price)
    
    # 정규화된 값으로 색상 구하기 (파란색 -> 빨간색)
    color = plt.cm.coolwarm(normalized_price)
    color_hex = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))
    
    # 지도에 원 추가
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=color_hex,
        fill=True,
        fill_color=color_hex,
        fill_opacity=0.7,
        popup=f"Price: ${row['Sale_Price']:,.0f}\nNeighborhood: {row['Neighborhood']}"
    ).add_to(ames_map)

# 범례 추가
colormap.add_to(ames_map)

# 결과를 HTML 파일로 저장
ames_map.save('../maps/ames_house_price_map_with_legend.html')

================================================================
import pandas as pd
import folium
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
# Ames 데이터 불러오기
df = pd.read_csv("../Ames_Project/data/houseprice-with-lonlat.csv")

# 가격 구간별 집 갯수
# Define the bin edges for the price ranges (0 to max_price with 50,000 intervals)
bins = np.arange(0, df['Sale_Price'].max() + 50000, 50000)

# Calculate the number of houses in each price range
price_counts = pd.cut(df['Sale_Price'], bins=bins).value_counts().sort_index()

# Adjusting the tick labels to display more neatly and highlighting the 10~15만 range
plt.figure(figsize=(10, 6))
colors = ['skyblue'] * len(price_counts)  # Default color for all bars

# Finding the index for the 100,000 to 150,000 range
highlight_index = price_counts.index.get_loc(pd.Interval(100000, 150000, closed='right'))
colors[highlight_index] = 'red'  # Change color for 10~15만 range

# Create the bar plot with specified colors
plt.rcParams.update({'font.family' : 'Malgun Gothic'})
price_counts.plot(kind='bar', color=colors)
plt.title('가격 구간별 집의 갯수')
plt.xlabel('가격 구간 ($)')
plt.ylabel('집의 갯수')
plt.xticks(rotation=60, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
================================================================
#| title: Best GrLivArea in map

# 데이터 불러오기 
house= pd.read_csv("../../Ames_Project/data/houseprice-with-lonlat.csv")

# 사용 변수만 추출하기 
house = house[["Longitude","Latitude",'Gr_Liv_Area','Overall_Cond','Garage_Cars','Year_Built','Neighborhood','Year_Remod_Add','Sale_Price']]

# 100000 이상 150000 미만 집값 추출
house1015 = house[(house["Sale_Price"] >= 100000) & (house["Sale_Price"] < 150000)]

# 정렬하기 
house1015 = house1015.sort_values("Gr_Liv_Area", ascending = False)

# 평당 집값 변수 생성
house1015['Price_Per_Area'] = house1015['Sale_Price'] / house1015['Gr_Liv_Area']

# 평당 집값 기준으로 평가 변수 부여
house1015['Score_GrLivArea'] = np.where(house1015['Price_Per_Area'] <= house1015['Price_Per_Area'].quantile(0.2), 5,
                          np.where(house1015['Price_Per_Area'] <= house1015['Price_Per_Area'].quantile(0.4), 4,
                          np.where(house1015['Price_Per_Area'] <= house1015['Price_Per_Area'].quantile(0.6), 3,
                          np.where(house1015['Price_Per_Area'] <= house1015['Price_Per_Area'].quantile(0.8), 2,
                          1))))

# GrLivArea 지도 표시
from folium import Marker
from folium.plugins import MarkerCluster

# 기본 지도를 생성합니다.
m = folium.Map(location=[house1015['Latitude'].mean(),
                            house1015['Longitude'].mean()], 
                            zoom_start=12,
                            tiles="cartodbpositron")

# 마커 클러스터 생성
marker_cluster = MarkerCluster().add_to(m)

# 점수에 따른 색상 맵핑
score_color_mapping = {
    1: 'red',
    2: 'orange',
    3: 'yellow',
    4: 'green',
    5: 'blue'
}

# house1015 데이터프레임을 반복하여 마커 추가
for _, row in house1015.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=score_color_mapping[row['Score_GrLivArea']],
        fill=True,
        fill_color=score_color_mapping[row['Score_GrLivArea']],
        fill_opacity=0.7,
        popup=f"Score: {row['Score_GrLivArea']}<br>Price: ${row['Sale_Price']:,.0f}<br>Area: {row['Gr_Liv_Area']} sqft"
    ).add_to(marker_cluster)  # 생성된 마커를 바로 클러스터에 추가
    
# 지도를 저장하거나 표시합니다.
m.save("../maps/house_gla.html")
================================================================

# 2.Garage_Cars
#| title: Best GarageCars in map
m2 = folium.Map(location=[house1015['Latitude'].mean(),
                            house1015['Longitude'].mean()], 
                            zoom_start=12,
                            tiles="cartodbpositron")

# 마커 클러스터 생성
marker_cluster = MarkerCluster().add_to(m2)

# 점수에 따른 색상 맵핑
score_color_mapping = {
    1: 'red',
    2: 'orange',
    3: 'yellow',
    4: 'green',
    5: 'blue'
}

house1015['Score_GarageCars'] = np.where(house1015['Garage_Cars'] == 0, 1,
                             np.where((house1015['Garage_Cars'] >= 1) & (house1015['Garage_Cars'] <= 2), 3,
                             np.where((house1015['Garage_Cars'] >= 3) & (house1015['Garage_Cars'] <= 5), 5, None)))
house1015[['Score_GarageCars']] 

# house1015 데이터프레임을 반복하여 마커
for _, row in house1015.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=score_color_mapping[row['Score_GarageCars']],
        fill=True,
        fill_color=score_color_mapping[row['Score_GarageCars']],
        fill_opacity=0.7,
        popup=f"Score: {row['Score_GarageCars']}<br>Price: ${row['Sale_Price']:,.0f}<br>Garage_Cars: {row['Garage_Cars']}"
    ).add_to(marker_cluster)

# 지도를 저장
m2.save("../maps/house_car.html")


================================================================
#| title: Best Overall_Cond in map
m3 = folium.Map(location=[house1015['Latitude'].mean(),
                            house1015['Longitude'].mean()], 
                            zoom_start=12,
                            tiles="cartodbpositron")

# 마커 클러스터 생성
marker_cluster = MarkerCluster().add_to(m3)

# 숫자형으로 변경해 평가 변수 부여
house1015[["Overall_Cond"]]
house1015["Score_Overall_Cond"] = np.where(house1015["Overall_Cond"] == 'Very_Poor', 1,
                                   np.where(house1015["Overall_Cond"] == "Poor", 1,
                                   np.where(house1015["Overall_Cond"] == "Fair", 2,
                                   np.where(house1015["Overall_Cond"] == "Below_Average", 2,
                                   np.where(house1015["Overall_Cond"] == "Average", 3,
                                   np.where(house1015["Overall_Cond"] == "Above_Average", 3,
                                   np.where(house1015["Overall_Cond"] == "Good", 4,
                                   np.where(house1015["Overall_Cond"] == "Very_Good", 4, 5))))))))
house1015[["Score_Overall_Cond"]]

# 점수에 따른 색상 맵핑
score_color_mapping = {
    1: 'red',
    2: 'orange',
    3: 'yellow',
    4: 'green',
    5: 'blue'
}

# house1015 데이터프레임을 반복하여 마커
for _, row in house1015.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=score_color_mapping[row['Score_Overall_Cond']],
        fill=True,
        fill_color=score_color_mapping[row['Score_Overall_Cond']],
        fill_opacity=0.7,
        popup=f"Score: {row['Score_Overall_Cond']}<br>Price: ${row['Sale_Price']:,.0f}<br>Overall_Cond: {row['Overall_Cond']}"
    ).add_to(marker_cluster)

# 지도를 저장
m3.save("../maps/house_cond.html")
================================================================

#| title: Best Year_Remod_Add in map

# 리모델링 연도에 따른 가성비 집
house[["Year_Remod_Add"]].max() #2010
house[["Year_Remod_Add"]].min() #1950

#각 집의 상태를 평가한 점수 부여
def calculate_condition_score(row):
    if row["Year_Remod_Add"] > 1998:
        return 5  # 리모델링이 최근에 이루어진 경우
    elif row["Year_Remod_Add"] > 1986:
        return 4
    elif row["Year_Remod_Add"] > 1974:
        return 3
    elif row["Year_Remod_Add"] > 1962:
        return 2
    else:
        return 1  # 오래된 주택일 경우


# 상태에 따른 평가 변수 추가
house1015["Score_year_remod"] = house1015.apply(calculate_condition_score, axis=1)
house1015[["Score_year_remod"]]
#| title: 01. 막대그래프 

# 지도 만들기 
# 기본 지도를 생성합니다.
m4 = folium.Map(location=[house1015['Latitude'].mean(),
                            house1015['Longitude'].mean()], 
                            zoom_start=12,
                            tiles="cartodbpositron")

# 마커 클러스터를 생성합니다.
marker_cluster = MarkerCluster().add_to(m4)

# 점수에 따른 색상 맵핑
score_color_mapping = {
    1: 'red',
    2: 'orange',
    3: 'yellow',
    4: 'green',
    5: 'blue'
}

# house1015 데이터프레임 반복, 마커 추가 
for _, row in house1015.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=score_color_mapping[row['Score_year_remod']],
        fill=True,
        fill_color=score_color_mapping[row['Score_year_remod']],
        fill_opacity=0.7,
        popup=f"Score: {row['Score_year_remod']}<br>Price: ${row['Sale_Price']:,.0f}<br>Year_Remod_Add: {row['Year_Remod_Add']}"
    ).add_to(marker_cluster)

# 지도를 저장
m4.save("../maps/house_remod.html")
================================================================

#| title: Best Year_Built in map

# 년도 별 점수 부여하기 
 ## 최대/최소값 확인
built_min = house1015['Year_Built'].min() # 1872
built_max = house1015['Year_Built'].max() # 2008
 ## 구간 구하기 
(built_max - built_min) / 5 # 27.2
np.arange(1872, 2009, 27.2)

x_1 = np.arange(1872, 1900)
x_2 = np.arange(1900, 1927)
x_3 = np.arange(1927, 1954)
x_4 = np.arange(1954, 1981)
x_5 = np.arange(1981, 2009)

# 평가 변수 부여하기 
## 방법 1
house1015["Score_Year_Built"] = np.where(house1015["Year_Built"].isin(x_1), 1,
                            np.where(house1015["Year_Built"].isin(x_2), 2,
                            np.where(house1015["Year_Built"].isin(x_3), 3,
                            np.where(house1015["Year_Built"].isin(x_4), 4, 5
                            ))))

house1015[["Score_Year_Built"]]

# 지도 만들기 
# 기본 지도를 생성합니다.
m5 = folium.Map(location=[house1015['Latitude'].mean(),
                            house1015['Longitude'].mean()], 
                            zoom_start=12,
                            tiles="cartodbpositron")

# 마커 클러스터를 생성합니다.
marker_cluster = MarkerCluster().add_to(m5)

# 점수에 따른 색상 맵핑
score_color_mapping = {
    1: 'red',
    2: 'orange',
    3: 'yellow',
    4: 'green',
    5: 'blue'
}

# house1015 데이터프레임 반복, 마커 추가 
for _, row in house1015.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        fill=True,
        fill_color=score_color_mapping[row['Score_Year_Built']],
        fill_opacity=0.7,
        popup=f"Score: {row['Score_Year_Built']}<br>Price: ${row['Sale_Price']:,.0f}<br>Year Built: {row['Year_Built']}"
    ).add_to(marker_cluster)

# 지도를 저장
m5.save("../maps/house_built.html")
=====================================================

# 종합평가 부분

# 종합 평가 점수 계산
house1015['Total_Score'] = (
    house1015['Score_GrLivArea'] + 
    house1015['Score_GarageCars'] + 
    house1015['Score_Overall_Cond'] + 
    house1015['Score_year_remod'] + 
    house1015['Score_Year_Built']) / 5

house1015

pip install shapely
from shapely.geometry import Point, Polygon, LineString

pip install geopandas
import geopandas as gpd


my_map = folium.Map(location=[42.034722, -93.62],
                     zoom_start=12,
                     tiles='cartodbpositron')

# 동네별 가운데 위치
center = house.groupby('Neighborhood')[['Longitude', 'Latitude']].median()
center

# 인덱스와 좌표를 추출
neighborhoods = center.index
longitudes = center['Longitude']
latitudes = center['Latitude']


## 지도2
# zip을 사용하여 텍스트 추가 (only text)
for neighborhood, lon, lat in zip(neighborhoods, longitudes, latitudes):
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black;">{neighborhood}</div>'),
        icon_size=(0, 0)
    ).add_to(my_map)
    
# 맵 저장
#my_map.save("map_with_text.html")

#----------------------------------------------------------------------

for neighborhood, lon, lat in zip(neighborhoods, longitudes, latitudes):
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black;">{neighborhood}</div>'),
        icon_size=(0, 0)
    ).add_to(my_map)

# 2. GeoDataFrame 생성
geometry = [Point(xy) for xy in zip(house['Longitude'], house['Latitude'])]
geo_df = gpd.GeoDataFrame(house, geometry=geometry)

# 3. 동네별로 Convex Hull 생성
neighborhoods = geo_df['Neighborhood'].unique()

# 빈 리스트 생성
boundary_polygons = []

for neighborhood in neighborhoods:
    # 동네별 데이터 필터링
    df_neigh = geo_df[geo_df['Neighborhood'] == neighborhood]
    
    # Convex Hull 생성
    if len(df_neigh) > 2:  # Convex Hull은 3개 이상의 점이 필요
        convex_hull = df_neigh.geometry.union_all().convex_hull
        boundary_polygons.append((neighborhood, convex_hull))

# 4. folium 지도 생성
#my_map = folium.Map(location=[geo_df.geometry.y.mean(), geo_df.geometry.x.mean()], zoom_start=12,tiles='cartodbpositron')

# 5. 경계 그리기
for neighborhood, poly in boundary_polygons:
    folium.Polygon(locations=[(point[1], point[0]) for point in poly.exterior.coords],
                   color='blue', weight=2, fill=True, fill_opacity=0.1).add_to(my_map)

geometry = [Point(xy) for xy in zip(house1015['Longitude'], house1015['Latitude'])]
geo_df = gpd.GeoDataFrame(house1015, geometry=geometry)

# 3. 구간 설정 및 색깔 매핑 함수 정의
def get_color(score):
    if score >= 4:
        return 'red'  # 4점 이상
    elif score >= 3:
        return 'orange'  # 4점 미만 3점 이상
    elif score >= 2:
        return 'yellow'  # 3점 미만 2점 이상
    elif score >= 1:
        return 'green'  # 2점 미만 1점 이상
    else:
        return 'blue'  # 1점 미만

# 4. 집 위치에 따른 마커 생성 및 색깔 지정
for _, row in geo_df.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=7,
        color=get_color(row['Total_Score']),
        fill=True,
        fill_color=get_color(row['Total_Score']),
        fill_opacity=0.7,
        popup=f"Score: {row['Total_Score']}"
    ).add_to(my_map)

# 5. 맵 저장 (기존 my_map에 추가된 상태로 저장)
my_map.save("../maps/map_with_final.html")

=====================================================
# ===================================
# 변수별로 가치를 다르게 둘때
# ===================================

# --------------------------------------------------
# Score_Overall_Cond
# --------------------------------------------------
my_map = folium.Map(location=[42.034722, -93.62],
                     zoom_start=12,
                     tiles='cartodbpositron')

# 동네별 가운데 위치
center = house.groupby('Neighborhood')[['Longitude', 'Latitude']].median()
center

# 인덱스와 좌표를 추출
neighborhoods = center.index
longitudes = center['Longitude']
latitudes = center['Latitude']

## 지도2
# zip을 사용하여 텍스트 추가 (only text)
for neighborhood, lon, lat in zip(neighborhoods, longitudes, latitudes):
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: blue;">{neighborhood}</div>'),
        icon_size=(0, 0)
    ).add_to(my_map)

# 맵 저장
my_map.save("../maps/map_with_text.html")

for neighborhood, lon, lat in zip(neighborhoods, longitudes, latitudes):
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black;">{neighborhood}</div>'),
        icon_size=(0, 0)
    ).add_to(my_map)

# 2. GeoDataFrame 생성
geometry = [Point(xy) for xy in zip(house['Longitude'], house['Latitude'])]
geo_df = gpd.GeoDataFrame(house, geometry=geometry)

# 3. 동네별로 Convex Hull 생성
neighborhoods = geo_df['Neighborhood'].unique()

# 빈 리스트 생성
boundary_polygons = []

for neighborhood in neighborhoods:
    # 동네별 데이터 필터링
    df_neigh = geo_df[geo_df['Neighborhood'] == neighborhood]
    
    # Convex Hull 생성
    if len(df_neigh) > 2:  # Convex Hull은 3개 이상의 점이 필요
        convex_hull = df_neigh.geometry.union_all().convex_hull
        boundary_polygons.append((neighborhood, convex_hull))

# 4. folium 지도 생성
#my_map = folium.Map(location=[geo_df.geometry.y.mean(), geo_df.geometry.x.mean()], zoom_start=12,tiles='cartodbpositron')

# 5. 경계 그리기
for neighborhood, poly in boundary_polygons:
    folium.Polygon(locations=[(point[1], point[0]) for point in poly.exterior.coords],
                   color='blue', weight=2, fill=True, fill_opacity=0.1).add_to(my_map)

# 6. 맵 저장
# my_map.save("../maps/my_map.html")

house1015_Overall = house1015.sort_values(by=['Score_Overall_Cond', 'Total_Score',"Sale_Price", 'Neighborhood'], ascending=False).reset_index(drop=True).head(10)
house1015_Overall[["Score_Overall_Cond",'Total_Score',"Sale_Price", 'Neighborhood']]

# 상위 10개 집 위치에 마커 추가
for idx, row in house1015_Overall.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],  # 위치를 위도와 경도로 설정
        popup=f"Price: ${row['Sale_Price']:,}<br>Neighborhood: {row['Neighborhood']}<br>TotalScore: {row['Total_Score']}<br>Score: {row['Score_Overall_Cond']:.2f}"
    ).add_to(my_map)
    
# 지도를 HTML 파일로 저장하거나 노트북에서 바로 표시
my_map.save("../maps/house1015_Overall.html")


# csv로 저장
df = house1015_Overall[["Longitude", 'Latitude', 'Neighborhood','Sale_Price', 'Score_GrLivArea', 'Score_Overall_Cond',
                                    'Score_GarageCars', 'Score_year_remod','Score_Year_Built', 'Total_Score']]

df.to_csv('../data/house1015_Overall.csv', index=False)


# --------------------------------------------------
# Score_GrLivArea
# --------------------------------------------------
my_map2 = folium.Map(location=[42.034722, -93.62],
                     zoom_start=12,
                     tiles='cartodbpositron')

# 동네별 가운데 위치
center = house.groupby('Neighborhood')[['Longitude', 'Latitude']].median()
center

# 인덱스와 좌표를 추출
neighborhoods = center.index
longitudes = center['Longitude']
latitudes = center['Latitude']

## 지도2
# zip을 사용하여 텍스트 추가 (only text)
for neighborhood, lon, lat in zip(neighborhoods, longitudes, latitudes):
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: blue;">{neighborhood}</div>'),
        icon_size=(0, 0)
    ).add_to(my_map2)

# 맵 저장
my_map2.save("../maps/map2_with_text.html")

for neighborhood, lon, lat in zip(neighborhoods, longitudes, latitudes):
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black;">{neighborhood}</div>'),
        icon_size=(0, 0)
    ).add_to(my_map2)

# 2. GeoDataFrame 생성
geometry = [Point(xy) for xy in zip(house['Longitude'], house['Latitude'])]
geo_df = gpd.GeoDataFrame(house, geometry=geometry)

# 3. 동네별로 Convex Hull 생성
neighborhoods = geo_df['Neighborhood'].unique()

# 빈 리스트 생성
boundary_polygons = []

for neighborhood in neighborhoods:
    # 동네별 데이터 필터링
    df_neigh = geo_df[geo_df['Neighborhood'] == neighborhood]
    
    # Convex Hull 생성
    if len(df_neigh) > 2:  # Convex Hull은 3개 이상의 점이 필요
        convex_hull = df_neigh.geometry.union_all().convex_hull
        boundary_polygons.append((neighborhood, convex_hull))

# 4. folium 지도 생성
#my_map = folium.Map(location=[geo_df.geometry.y.mean(), geo_df.geometry.x.mean()], zoom_start=12,tiles='cartodbpositron')

# 5. 경계 그리기
for neighborhood, poly in boundary_polygons:
    folium.Polygon(locations=[(point[1], point[0]) for point in poly.exterior.coords],
                   color='blue', weight=2, fill=True, fill_opacity=0.1).add_to(my_map2)

# 6. 맵 저장
# my_map2.save("../maps/my_map2.html")


house1015_GrLivArea = house1015.sort_values(by=['Score_GrLivArea', 'Total_Score',"Sale_Price"], ascending=False).reset_index(drop=True).head(10)
house1015_GrLivArea[["Score_GrLivArea",'Total_Score',"Sale_Price", 'Neighborhood']]


# 상위 10개 집 위치에 마커 추가
for idx, row in house1015_GrLivArea.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],  # 위치를 위도와 경도로 설정
        popup=f"Price: ${row['Sale_Price']:,}<br>Neighborhood: {row['Neighborhood']}<br>TotalScore: {row['Total_Score']}<br>Score: {row['Score_GrLivArea']:.2f}"
    ).add_to(my_map2)

# 지도를 HTML 파일로 저장하거나 노트북에서 바로 표시
my_map2.save("../maps/house1015_GrLivArea.html")

# csv로 저장
df = house1015_GrLivArea[["Longitude", 'Latitude', 'Neighborhood','Sale_Price', 'Score_GrLivArea', 'Score_Overall_Cond',
                                    'Score_GarageCars', 'Score_year_remod','Score_Year_Built', 'Total_Score']]

df.to_csv('../data/house1015_GrLivArea.csv', index=False)

# --------------------------------------------------
# Score_year_remod
# --------------------------------------------------
my_map3 = folium.Map(location=[42.034722, -93.62],
                     zoom_start=12,
                     tiles='cartodbpositron')

# 동네별 가운데 위치
center = house.groupby('Neighborhood')[['Longitude', 'Latitude']].median()
center

# 인덱스와 좌표를 추출
neighborhoods = center.index
longitudes = center['Longitude']
latitudes = center['Latitude']

## 지도2
# zip을 사용하여 텍스트 추가 (only text)
for neighborhood, lon, lat in zip(neighborhoods, longitudes, latitudes):
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: blue;">{neighborhood}</div>'),
        icon_size=(0, 0)
    ).add_to(my_map3)

# 맵 저장
my_map3.save("../maps/map3_with_text.html")

for neighborhood, lon, lat in zip(neighborhoods, longitudes, latitudes):
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black;">{neighborhood}</div>'),
        icon_size=(0, 0)
    ).add_to(my_map3)

# 2. GeoDataFrame 생성
geometry = [Point(xy) for xy in zip(house['Longitude'], house['Latitude'])]
geo_df = gpd.GeoDataFrame(house, geometry=geometry)

# 3. 동네별로 Convex Hull 생성
neighborhoods = geo_df['Neighborhood'].unique()

# 빈 리스트 생성
boundary_polygons = []

for neighborhood in neighborhoods:
    # 동네별 데이터 필터링
    df_neigh = geo_df[geo_df['Neighborhood'] == neighborhood]
    
    # Convex Hull 생성
    if len(df_neigh) > 2:  # Convex Hull은 3개 이상의 점이 필요
        convex_hull = df_neigh.geometry.union_all().convex_hull
        boundary_polygons.append((neighborhood, convex_hull))

# 5. 경계 그리기
for neighborhood, poly in boundary_polygons:
    folium.Polygon(locations=[(point[1], point[0]) for point in poly.exterior.coords],
                   color='blue', weight=2, fill=True, fill_opacity=0.1).add_to(my_map3)


house1015_year_remod = house1015.sort_values(by=['Score_year_remod', 'Total_Score',"Sale_Price"], ascending=False).reset_index(drop=True).head(10)
house1015_year_remod[["Score_year_remod",'Total_Score',"Sale_Price", 'Neighborhood']]


# 상위 10개 집 위치에 마커 추가
for idx, row in house1015_year_remod.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],  # 위치를 위도와 경도로 설정
         popup=f"Price: ${row['Sale_Price']:,}<br>Neighborhood: {row['Neighborhood']}<br>TotalScore: {row['Total_Score']}<br>Score: {row['Score_year_remod']:.2f}"
    ).add_to(my_map3)

# 지도를 HTML 파일로 저장하거나 노트북에서 바로 표시
my_map3.save("../maps/house1015_year_remod.html")

# csv로 저장
df = house1015_year_remod[["Longitude", 'Latitude', 'Neighborhood','Sale_Price', 'Score_GrLivArea', 'Score_Overall_Cond',
                                    'Score_GarageCars', 'Score_year_remod','Score_Year_Built', 'Total_Score']]

df.to_csv('../data/house1015_year_remod.csv', index=False)


# -------------------------------------------------
# Score_Year_Built
# -------------------------------------------------
my_map4 = folium.Map(location=[42.034722, -93.62],
                     zoom_start=12,
                     tiles='cartodbpositron')

# 동네별 가운데 위치
center = house.groupby('Neighborhood')[['Longitude', 'Latitude']].median()
center

# 인덱스와 좌표를 추출
neighborhoods = center.index
longitudes = center['Longitude']
latitudes = center['Latitude']

## 지도2
# zip을 사용하여 텍스트 추가 (only text)
for neighborhood, lon, lat in zip(neighborhoods, longitudes, latitudes):
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: blue;">{neighborhood}</div>'),
        icon_size=(0, 0)
    ).add_to(my_map4)

# 맵 저장
my_map4.save("../maps/map4_with_text.html")

for neighborhood, lon, lat in zip(neighborhoods, longitudes, latitudes):
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black;">{neighborhood}</div>'),
        icon_size=(0, 0)
    ).add_to(my_map4)

# 2. GeoDataFrame 생성
geometry = [Point(xy) for xy in zip(house['Longitude'], house['Latitude'])]
geo_df = gpd.GeoDataFrame(house, geometry=geometry)

# 3. 동네별로 Convex Hull 생성
neighborhoods = geo_df['Neighborhood'].unique()

# 빈 리스트 생성
boundary_polygons = []

for neighborhood in neighborhoods:
    # 동네별 데이터 필터링
    df_neigh = geo_df[geo_df['Neighborhood'] == neighborhood]
    
    # Convex Hull 생성
    if len(df_neigh) > 2:  # Convex Hull은 3개 이상의 점이 필요
        convex_hull = df_neigh.geometry.union_all().convex_hull
        boundary_polygons.append((neighborhood, convex_hull))

# 5. 경계 그리기
for neighborhood, poly in boundary_polygons:
    folium.Polygon(locations=[(point[1], point[0]) for point in poly.exterior.coords],
                   color='blue', weight=2, fill=True, fill_opacity=0.1).add_to(my_map4)



house1015_Year_Built = house1015.sort_values(by=['Score_Year_Built', 'Total_Score',"Sale_Price"], ascending=False).reset_index(drop=True).head(10)
house1015_Year_Built[["Score_Year_Built",'Total_Score',"Sale_Price", 'Neighborhood']]

## 지도
# 지도의 중심 위치를 결정하기 위해 중간값을 사용
map_center = [house1015_year_remod['Latitude'].mean(), house1015_year_remod['Longitude'].mean()]


# 상위 10개 집 위치에 마커 추가
for idx, row in house1015_Year_Built.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],  # 위치를 위도와 경도로 설정
        popup=f"Price: ${row['Sale_Price']:,}<br>Neighborhood: {row['Neighborhood']}<br>TotalScore: {row['Total_Score']}<br>Score: {row['Score_Year_Built']:.2f}"
    ).add_to(my_map4)

# 지도를 HTML 파일로 저장하거나 노트북에서 바로 표시
my_map4.save("../maps/house1015_Year_Built.html")

# csv로 저장
df = house1015_Year_Built[["Longitude", 'Latitude', 'Neighborhood','Sale_Price', 'Score_GrLivArea', 'Score_Overall_Cond',
                                    'Score_GarageCars', 'Score_year_remod','Score_Year_Built', 'Total_Score']]

df.to_csv('../data/house1015_Year_Built.csv', index=False)

=======================================================
#| title: 2. Ames시의 가성비 Top 6

top_value_houses = house1015[["Longitude", 'Latitude', 'Neighborhood','Sale_Price', 'Score_GrLivArea', 'Score_Overall_Cond',
                                    'Score_GarageCars', 'Score_year_remod','Score_Year_Built', 'Total_Score']]
top_value_houses

# 상위순으로 정렬 
top_value_houses = top_value_houses.sort_values(by='Total_Score', ascending=False)
top_value_houses

# 종합 점수 상위 6개 집의 데이터 선택
top_6_houses = top_value_houses.head(6)
pd.set_option('display.max_columns', 10)
top_6_houses

my_map = folium.Map(location=[42.034722, -93.62],
                     zoom_start=12,
                     tiles='cartodbpositron')

# 동네별 가운데 위치
center = house.groupby('Neighborhood')[['Longitude', 'Latitude']].mean()
center

# 인덱스와 좌표를 추출
neighborhoods = center.index
longitudes = center['Longitude']
latitudes = center['Latitude']

## 지도1
# zip을 사용하여 텍스트 추가 (only text)
for neighborhood, lon, lat in zip(neighborhoods, longitudes, latitudes):
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black;">{neighborhood}</div>'),
        icon_size=(0, 0)
    ).add_to(my_map)
    
# 맵 저장
my_map.save("../maps/map_with_text.html")

for neighborhood, lon, lat in zip(neighborhoods, longitudes, latitudes):
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black;">{neighborhood}</div>'),
        icon_size=(0, 0)
    ).add_to(my_map)

# 2. GeoDataFrame 생성
geometry = [Point(xy) for xy in zip(house['Longitude'], house['Latitude'])]
geo_df = gpd.GeoDataFrame(house, geometry=geometry)

# 3. 동네별로 Convex Hull 생성
neighborhoods = geo_df['Neighborhood'].unique()

# 빈 리스트 생성
boundary_polygons = []

for neighborhood in neighborhoods:
    # 동네별 데이터 필터링
    df_neigh = geo_df[geo_df['Neighborhood'] == neighborhood]
    
    # Convex Hull 생성
    if len(df_neigh) > 2:  # Convex Hull은 3개 이상의 점이 필요
        convex_hull = df_neigh.geometry.union_all().convex_hull
        boundary_polygons.append((neighborhood, convex_hull))

# 4. folium 지도 생성
#my_map = folium.Map(location=[geo_df.geometry.y.mean(), geo_df.geometry.x.mean()], zoom_start=12,tiles='cartodbpositron')

# 각 상위 6개 집에 마커 추가
for _, row in top_6_houses.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],  # 'Latitude'와 'Longitude' 컬럼을 사용하는 것으로 가정
        radius=7,  # 원의 반경
        color='red',  # 외곽선 색상
        fill=True,
        fill_color='red',  # 채우기 색상
        fill_opacity=0.7,  # 채우기 투명도
        popup=(
            f"Neighborhood: {row['Neighborhood']}<br>"
            f"Sale Price: ${row['Sale_Price']:,}<br>"
            f"Total Score: {row['Total_Score']:.2f}"
        )
    ).add_to(my_map)


# 지도를 HTML 파일로 저장하거나 IPython 환경에서 표시
my_map.save("../maps/top_6_houses.html")

# 5. 경계 그리기
for neighborhood, poly in boundary_polygons:
    folium.Polygon(locations=[(point[1], point[0]) for point in poly.exterior.coords],
                   color='blue', weight=2, fill=True, fill_opacity=0.1).add_to(my_map)

# 6. 맵 저장
my_map.save("../maps/my_map_top6.html")

====================================================
# 동네별 상위 2곳
my_map = folium.Map(location=[42.034722, -93.62],
                     zoom_start=12,
                     tiles='cartodbpositron')
                     

# 동네별 가운데 위치
center = house.groupby('Neighborhood')[['Longitude', 'Latitude']].median()
center

# 인덱스와 좌표를 추출
neighborhoods = center.index
longitudes = center['Longitude']
latitudes = center['Latitude']


## 지도2
# zip을 사용하여 텍스트 추가 (only text)
for neighborhood, lon, lat in zip(neighborhoods, longitudes, latitudes):
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black;">{neighborhood}</div>'),
        icon_size=(0, 0)
    ).add_to(my_map)
    
# 맵 저장
my_map.save("../maps/map_with_text.html")

#----------------------------------------------------------------------

#!pip install shapely
from shapely.geometry import Point, Polygon, LineString

for neighborhood, lon, lat in zip(neighborhoods, longitudes, latitudes):
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black;">{neighborhood}</div>'),
        icon_size=(0, 0)
    ).add_to(my_map)

# 2. GeoDataFrame 생성
#!pip install geopandas
import geopandas as gpd

geometry = [Point(xy) for xy in zip(house['Longitude'], house['Latitude'])]
geo_df = gpd.GeoDataFrame(house, geometry=geometry)

# 3. 동네별로 Convex Hull 생성
neighborhoods = geo_df['Neighborhood'].unique()

# 빈 리스트 생성
boundary_polygons = []

for neighborhood in neighborhoods:
    # 동네별 데이터 필터링
    df_neigh = geo_df[geo_df['Neighborhood'] == neighborhood]
    
    # Convex Hull 생성
    if len(df_neigh) > 2:  # Convex Hull은 3개 이상의 점이 필요
        convex_hull = df_neigh.geometry.union_all().convex_hull
        boundary_polygons.append((neighborhood, convex_hull))

# 4. folium 지도 생성
#my_map = folium.Map(location=[geo_df.geometry.y.mean(), geo_df.geometry.x.mean()], zoom_start=12,tiles='cartodbpositron')

# 5. 경계 그리기
for neighborhood, poly in boundary_polygons:
    folium.Polygon(locations=[(point[1], point[0]) for point in poly.exterior.coords],
                   color='blue', weight=2, fill=True, fill_opacity=0.1).add_to(my_map)

# 6. 맵 저장
my_map.save("../maps/my_map.html")

grouped = house1015.groupby('Neighborhood')

# 각 그룹 내에서 Total_Score 기준으로 정렬한 후 상위 2개의 행만 추출
top_2_per_group = grouped.apply(lambda x: x.sort_values(by='Total_Score', ascending=False).head(2))

for _, row in top_2_per_group.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,  # 원의 반경
        color='red',  # 외곽선 색상
        fill=True,
        fill_color='red',  # 채우기 색상
        fill_opacity=0.7,  # 채우기 투명도
        popup=(
            f"Neighborhood: {row['Neighborhood']}<br>"
            f"Sale Price: ${row['Sale_Price']:,}<br>"
            f"Total Score: {row['Total_Score']:.2f}"
        )
    ).add_to(my_map)

# 지도를 HTML 파일로 저장하거나 IPython 환경에서 표시
my_map.save("../maps/top_2_houses_map.html")

# csv로 저장
df = top_2_per_group[["Longitude", 'Latitude', 'Neighborhood','Sale_Price', 'Score_GrLivArea', 'Score_Overall_Cond',
                                    'Score_GarageCars', 'Score_year_remod','Score_Year_Built', 'Total_Score']]

df.to_csv('../data/top_2_per_group.csv', index=False)
