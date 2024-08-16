import json
geo_seoul = json.load(open("data/SIG_Seoul.geojson", encoding = "UTF-8"))
geo_seoul


type(geo_seoul)
len(geo_seoul)
geo_seoul.keys()
geo_seoul["name"]
geo_seoul["features"][0]
len(geo_seoul["features"]) # 구의 갯수 = 25
len(geo_seoul["features"][0])
geo_seoul["features"][0].keys()

# 행정 구역 코드 출력
# 숫자가 바뀌면 "구"가 바뀌는 구나!
geo_seoul["features"][0]["properties"]
geo_seoul["features"][2]["properties"]

# 위도, 경도 좌표 출력
len(geo_seoul["features"][0]["geometry"])
geo_seoul["features"][0]["geometry"].keys()
coordinate_list = geo_seoul["features"][0]["geometry"]["coordinates"]
len(coordinate_list[0][0])
coordinate_list[0][0]

import numpy as np
coordinate_array = np.array(coordinate_list[0][0])
x = coordinate_array[:,0]
y = coordinate_array[:,1]

import matplotlib.pyplot as plt
plt.plot(x[::10], y[::10])
plt.show()
plt.clf()

# 함수로 만들기 
def draw_seoul(num) :
    gu_name = geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list = geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array = np.array(coordinate_list[0][0])
    x = coordinate_array[:,0]
    y = coordinate_array[:,1]

    plt.rcParams.update({"font.family" : "Malgun Gothic"})
    plt.plot(x, y)
    plt.title(gu_name)
    # 축 비율 1:1로 설정
    plt.axis("equal")
    plt.show()
    plt.clf()    
    
    return None

draw_seoul(21)

# 서울시 전체 지도 그리기
# gu_name | x | y
# ===============
# 종로구  | 126 | 36
# 종로구  | 126 | 36
# 종로구  | 126 | 36
# ......
# 종로구  | 126 | 36
# 종로구  | 126 | 36
# 중구  | 126 | 36
# 중구  | 126 | 36
# ......
# 중구  | 126 | 36
# x, y 판다스 데이터 프레임
import pandas as pd

 def make_seouldf(num) :
       gu_name = geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
       coordinate_list = geo_seoul["features"][num]["geometry"]["coordinates"]
       coordinate_array = np.array(coordinate_list[0][0])
       x = coordinate_array[:,0]
       y = coordinate_array[:,1]
   
       return pd.DataFrame({"gu_name" : gu_name,
                            "x" : x,
                            "y" : y})

make_seouldf(14)

result = pd.DataFrame({})
for i in range(25) :
    result= pd.concat([result, make_seouldf(i)], ignore_index = True)

result

import seaborn as sns

sns.scatterplot(data = result,
                x = "x", y = "y", hue = "gu_name", s = 2,
                legend = False)
plt.show()
plt.clf()

# 다른 방법

def df_sh(i):
    coordinate_list = geo_seoul["features"][i]["geometry"]["coordinates"]
    coordinate_array = np.array(coordinate_list[0][0])
    gu_name = [geo_seoul["features"][i]["properties"]["SIG_KOR_NM"]]
    df = pd.DataFrame({"x" : coordinate_array[:,0],
                       "y" : coordinate_array[:,1]})
    df["gu_name"] = gu_name * len(coordinate_array)
    df = df[["gu_name","x","y"]]
    return df

df_sh(1)

result = pd.DataFrame({})
for x in range(len(geo_seoul["features"])):
    result = pd.concat([result,df_sh(x)])
result = result.reset_index(drop=False) # 인덱스 꺼내기 or 없애기
# 칼럼명 index를 Id로 변환
# result.rename(columns = {"index" : "Id"})

import seaborn as sns

sns.scatterplot(data = result,
                x = "x", y = "y", hue = "gu_name", s = 2,
                legend = False)
plt.show()
plt.clf()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 구이름 만들기
# 방법 1
gu_name = []
for i in range(25) :
    gu_name.append(geo_seoul["features"][i]["properties"]["SIG_KOR_NM"])

# 방법 2 리스트 컨프리헨션
# gu_name = [geo_seoul["features"][i]["properties"]["SIG_KOR_NM"] for i in range(25)]
# gu_name 


# 서울 그래프 그리기(강남 강조)
import pandas as pd
gangnam_df = result.assign(is_gangnam = np.where(result["gu_name"] == "강남구", "강남", "안강남"))

sns.scatterplot(data = gangnam_df,
                x = "x", y = "y", palette = {"안강남" : "grey", "강남" : "red"}, hue = "is_gangnam", s = 2,
                legend = False)
plt.show()
plt.clf()



import numpy as np
import matplotlib.pyplot as plt
import json

geo_seoul = json.load(open("data/SIG_Seoul.geojson", encoding = "UTF-8"))
geo_seoul["features"][0]["properties"]

df_pop = pd.read_csv("data/population_SIG.csv")
df_pop.head()
df_seoulpop = df_pop.iloc[1:26]
df_seoulpop["code"] = df_seoulpop["code"].astype(str)
df_seoulpop.info()

# 패키지 설치하기
# !pip install folium
import folium 

center_x = result["x"].mean()
center_y = result["y"].mean()

# p. 304
# 흰 도화지 맵 가져오기
map_sig = folium.Map(location = [37.551, 126.973], # 지도 중심 좌표표
                    zoom_start = 12, # 확대 단계
                    tiles = "cartodbpositron") # 지도 종류
map_sig.save("map_seoul.html")
# 코로플릿 사용해서 -구 경계선 그리기
# 코로플릿 with bins
# matplotlib 팔레트
# tab10, tab20, Set1, Paired, Accent, Dark2, Pastel1, hsv 
# seaborn 팔레트
# deep, muted, bright, pastel, dark, colorblind, viridis, inferno, magma, plasma

bins = list(df_seoulpop["pop"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))

folium.Choropleth(
    geo_data = geo_seoul,
    data = df_seoulpop,
    columns=("code", "pop"),
    fill_color = "viridis",
    bins = bins,
    key_on = "feature.properties.SIG_CD").add_to(map_sig)
    
map_sig.save("map_seoul.html")

# 점 찍는 법
# make_seouldf(0).iloc[:,1:3].mean()
folium.Marker([37.583744,126.983800 ], popup = "종로구").add_to(map_sig)
map_sig.save("map_seoul.html")    
