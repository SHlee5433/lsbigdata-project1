# 집값 시각화 코드
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# 데이터 불러오기
file_path = '../lsbigdata-project1/data/houseprice/houseprice-with-lonlat.csv'
house = pd.read_csv(file_path)

house["Sale_Price"].sort_values(ascending=False)
house[house["Sale_Price"] == 755000] # 1767 가장 비싼 집

# 가격대별 집 분포 
a = house[(house["Sale_Price"] >= 100000) & (house["Sale_Price"] < 150000)]
b = house[(house["Sale_Price"] >= 150000) & (house["Sale_Price"] < 200000)]
c = house[(house["Sale_Price"] >= 200000) & (house["Sale_Price"] < 250000)]
d = house[(house["Sale_Price"] >= 250000) & (house["Sale_Price"] < 300000)]
e = house[(house["Sale_Price"] >= 50000) & (house["Sale_Price"] < 100000)]

a # 1016
b # 801
c # 415
d # 223
e # 226



# Overall_Cond

# conditions = [
#     house["Overall_Cond"] == 'Very_Poor',
#     house["Overall_Cond"] == 'Poor',
#     house["Overall_Cond"] == 'Fair',
#     house["Overall_Cond"] == 'Below_Average',
#     house["Overall_Cond"] == 'Average',
#     house["Overall_Cond"] == 'Above_Average',
#     house["Overall_Cond"] == 'Good',
#     house["Overall_Cond"] == 'Very_Good',
#     house["Overall_Cond"] == 'Excellent',
#     house["Overall_Cond"] == 'Very_Excellent'
# ]
# 
# values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 
# # np.select 사용하여 새로운 컬럼 생성
# house["Overall_Cond2"] = np.select(conditions, values, default=np.nan)
# house["Overall_Cond2"]

house_df = house.query("Sale_Price >= 100000 & Sale_Price < 150000")
house_df["Overall_Cond"]
house_df["Score_Overall_Cond"] = np.where(house_df["Overall_Cond"] == 'Very_Poor', 1,
                                   np.where(house_df["Overall_Cond"] == "Poor", 1,
                                   np.where(house_df["Overall_Cond"] == "Fair", 2,
                                   np.where(house_df["Overall_Cond"] == "Below_Average", 2,
                                   np.where(house_df["Overall_Cond"] == "Average", 3,
                                   np.where(house_df["Overall_Cond"] == "Above_Average", 3,
                                   np.where(house_df["Overall_Cond"] == "Good", 4,
                                   np.where(house_df["Overall_Cond"] == "Very_Good", 4, 5
                                   ))))))))
house_df["Score_Overall_Cond"]

# 시각화
house_df["Sale_Price"] = house_df["Sale_Price"] / 10000
x = house_df["Score_Overall_Cond"]
y = house_df["Sale_Price"]

# 산점도 필요 x(그냥 해봄봄)
sns.scatterplot(x = x, y = y, data = house_df)
plt.rcParams.update({"font.family" : "Malgun Gothic"})    
plt.xlabel("점수")
plt.ylabel("집값")
plt.xticks(range(1, 6, 1))
plt.show()
plt.clf()

# 카운트 
sns.countplot(x = x, data = house_df)
plt.rcParams.update({"font.family" : "Malgun Gothic"})    
plt.title("10만 ~ 15만 달러 주택의 컨디션 점수의 빈도수")
plt.xlabel("점수")
plt.ylabel("빈도")
plt.show()
plt.clf()


# 점수에 따른 집값
# 선그래프
sns.lineplot(x = x, y = y, data = house_df, errorbar = None, marker = "o")
plt.rcParams.update({"font.family" : "Malgun Gothic"})    
plt.title("10만 ~ 15만 달러 주택의 집값별 컨디션 점수")
plt.xticks(range(1, 6, 1))
plt.yticks(range(10, 15, 1))  
plt.xlabel("점수")
plt.ylabel("집값(만)")
plt.subplots_adjust(left=0.15, bottom=0.13)  # 여백 값은 필요에 맞게 조정 가능
plt.show()
plt.clf()


# 건축년도 점수
built_min = house_df['Year_Built'].min() # 1872
built_max = house_df['Year_Built'].max() # 2008

(built_max - built_min) / 5 # 27.2

np.arange(1872, 2009, 27.2)

# 년도 별 점수 부여하기 
x_1 = np.arange(1872, 1900)
x_2 = np.arange(1900, 1927)
x_3 = np.arange(1927, 1954)
x_4 = np.arange(1954, 1981)
x_5 = np.arange(1981, 2009)

# 변수 만들기
## 방법 1
house_df["Score_Year_Built"] = np.where(house_df["Year_Built"].isin(x_1), 1,
                            np.where(house_df["Year_Built"].isin(x_2), 2,
                            np.where(house_df["Year_Built"].isin(x_3), 3,
                            np.where(house_df["Year_Built"].isin(x_4), 4, 5
                            ))))

house_df["Score_Year_Built"]

## 방법 2
# bin_year = [1872, 1900, 1927, 1954, 1981, 2009]
# labels = [1, 2, 3, 4, 5] # bins의 갯수보다 1개 적어야 한다.
# house_df["Score_Year_Built"] = pd.cut(house_df["Year_Built"], bins = bin_year, 
#                                         labels = labels, right = False)



x1 = house_df["Score_Year_Built"]
y1 = house_df["Sale_Price"]

# 시각화
# 빈도 그래프 

sns.countplot(x = x1, data = house_df)
plt.rcParams.update({"font.family" : "Malgun Gothic"})    
plt.title("10만 ~ 15만 달러 주택의 건축년도 점수의 빈도수")
plt.xlabel("점수")
plt.ylabel("빈도")
plt.show()
plt.clf()

# 선그래프

sns.lineplot(x = x1, y = y1, data = house_df, errorbar = None, marker = "o")
plt.rcParams.update({"font.family" : "Malgun Gothic"})    
plt.title("10만 ~ 15만 달러 주택의 집값별 건축년도 점수")
plt.xticks(range(1, 6, 1))
plt.yticks(range(100000, 150000, 10000))
plt.xlabel("점수")
plt.ylabel("집값")
plt.show()
plt.clf()
