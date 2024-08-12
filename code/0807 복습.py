import json
geo_seoul = json.load(open("./data/SIG_Seoul.geojson", encoding = "UTF-8"))
geo_seoul


type(geo_seoul)
len(geo_seoul)
geo_seoul["features"][0].keys()

# 행정 구역 코드 출력
geo_seoul["features"][0]["properties"] 

gu_name = geo_seoul["features"][0]["properties"]["SIG_KOR_NM"] # 종로구

# 위도, 경도 좌표 출력
geo_seoul["features"][0]["geometry"]
coordinate_list = (geo_seoul["features"][0]["geometry"]["coordinates"]

x = np.array(coordinate_list[0][0])[:,0]
y = np.array(coordinate_list[0][0])[:,1]


jongro = pd.DataFrame({"gu_name" : gu_name,
                       "x" : x,
                       "y" : y})
              
jongro

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({"font.family" : "Malgun Gothic"})

plt.plot(x, y)
plt.title(gu_name)
plt.show()
plt.clf()

# 함수 만들어보기

def draw_seoul(num) :
    gu_name = geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list = geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array = np.array(coordinate_list[0][0])
    x = coordinate_array[:,0]
    y = coordinate_array[:,1]
    plt.rcParams.update({"font.family" : "Malgun Gothic"})
    plt.plot(x, y)
    plt.title(gu_name)
    plt.show()
    plt.clf()
    
    return None
    
draw_seoul(18)
    
# 서울시 전체 지도 그리기
    
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

# 구 이름 만들기
# 방법 1 리스트 컨프리헨션 한결사마 방법
gu_name = [geo_seoul["features"][i]["properties"]["SIG_KOR_NM"] for i in range(len(geo_seoul["features"]))]
    
# 방법 2 빈 리스트 생성 후 append 사용해서 채워넣기 용규갓 방법
gu_name = []
for i in range(len(geo_seoul["features"])) :
    gu_name.append(geo_seoul["features"][i]["properties"]["SIG_KOR_NM"])
    
gu_name
   
   
# 서울 그래프 그리기(강남 강조)
 
import pandas as pd
 
gangnam_df = result.assign(is_gangnam = np.where(gu_name == ["강남구"], "강남", "안강남"))
gangnam_df 

sns.scatterplot(data = gangnam_df, x = )


import numpy as np
np.array(1, 2, 4)
