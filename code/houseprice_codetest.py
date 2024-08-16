# 집값 시각화 코드
import pandas as pd

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
  for price, lat, lon in zip(house_df["Sale_Price"], house_df["Latitude"], house_df["Longitude"]):
      folium.Marker([lat,lon], popup = f"Price: ${price}").add_to(my_map)
      folium.Circle(
         location=[lat, lon],
         radius=50,
         fill=True,
         fill_opacity=0.6,
         popup=f"Price: ${price}",
     ).add_to(my_map) 

my_map.save('map_ames.html')

# 2번째 방법
# for idx, row in house_df.iterrows():
#     folium.Marker(
#         location=[house["Latitude"], house["Longitude"]],
#         popup="집들,,"
#     ).add_to(my_map)
    
# 3번쨰 방법
# for i in range(len(house_df["Latitude"])) :
#     folium.Marker([house_df[i:1], house_df[i:0]],
#     )
#     
#     for price, lat, lon in zip(house_df['Sale_Price'], house_df['Latitude'], house_df['Longitude']):
#     color = get_color(price)
#     folium.Circle(
#         location=[lat, lon],
#         radius=50,
#         color=color,
#         fill=True,
#         fill_opacity=0.6,
#         popup=f"Price: ${price}"
#     ).add_to(my_map)
# 집의 규모에 따라 원의 크기를 조절하면 좋음. 레디언스??
