# 집값 시각화 코드
import pandas as pd

# 데이터 불러오기
file_path = '../lsbigdata-project1/data/houseprice/houseprice-with-lonlat.csv'
data = pd.read_csv(file_path)

# 데이터의 기본 정보 확인
print(data.info())

# 누락된 값 확인
print(data.isna().sum())

# 위도와 경도의 기본 통계 확인
print(data[['Latitude', 'Longitude']].describe())

import plotly.express as px

# 데이터 프레임에서 위도와 경도 값이 있는지 확인
filtered_data = data.dropna(subset=['Latitude', 'Longitude'])

fig = px.scatter_mapbox(filtered_data, 
                        lat='Latitude', 
                        lon='Longitude', 
                        color='Sale_Price', 
                        size='Sale_Price',
                        color_continuous_scale=px.colors.cyclical.IceFire, 
                        size_max=15, 
                        zoom=10,
                        mapbox_style="carto-positron",
                        title='Ames의 집 가격')
fig.show()
