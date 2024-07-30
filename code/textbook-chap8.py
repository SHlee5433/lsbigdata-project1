import pandas as pd

mpg = pd.read_csv("data/mpg.csv")
mpg.shape

import seaborn as sns
import matplotlib.pyplot as plt

plt.clf()
# plt.figure(figsize=(5, 4)) # 사이즈 조정
sns.scatterplot(data = mpg,
                x= "displ", y= "hwy",
                hue = "drv") \
    .set(xlim=[3,6], ylim=[10, 30])
plt.show()

# 막대그래프
# mpg["drv"].unique()
df_mpg = mpg.groupby("drv", as_index=False) \
    .agg(mean_hwy = ("hwy", "mean"))
df_mpg
sns.barplot(data=df_mpg.sort_values("mean_hwy"),
            x = "drv" , y = "mean_hwy",
            hue = "drv")

plt.show()
plt.clf()
df_mpg = mpg.groupby("drv", as_index = False) \
    .agg(n = ("drv", "count"))
df_mpg    


sns.countplot(data = mpg, x ="drv", hue = "drv")
plt.show()


# 교재 8장, p.212
import seaborn as sns
import pandas as pd
economics = pd.read_csv("data/economics.csv")
economics.head()

economics.info()
sns.lineplot(data = economics, x = "date", y = "unemploy")
plt.show()
plt.clf()

economics["date2"] = pd.to_datetime(economics["date"])
economics.info()

economics[["date", "date2"]]
economics["date2"].dt.year
economics["date2"].dt.month
economics["date2"].dt.day
economics["date2"].dt.month_name()
economics["date2"].dt.quarter
economics["quarter"] = economics["date2"].dt.quarter
economics[["date2", "quarter"]]
# 각 날짜는 무슨 요일인가?
economics["date2"].dt.day_name()
economics["date2"] + pd.DateOffset(days=30)
economics["date2"] + pd.DateOffset(months=1)

economics["year"] = economics["date2"].dt.year
economics

sns.lineplot(data = economics, x = "year", y = "unemploy", errorbar = None)
sns.scatterplot(data = economics, x = "year", y = "unemploy", s =1)
plt.show()
plt.clf()

my_df = economics.groupby("year", as_index = False) \
         .agg(
             mon_mean=("unemploy", "mean"),
             mon_std=("unemploy", "std"),
             mon_n=("unemploy", "count")
         )
my_df 
mean + 1.96*std/sqrt(12)
my_df["left_ci"] = my_df["mon_mean"] - 1.96 * my_df["mon_std"] / np.sqrt(my_df["mon_n"])
my_df["right_ci"] = my_df["mon_mean"] + 1.96 * my_df["mon_std"] / np.sqrt(my_df["mon_n"])
my_df.head()

import matplotlib.pyplot as plt

x = my_df["year"]
y = my_df["mon_mean"]
# plt.scatter(x, y, s=3)
plt.plot(x, y, color = "black")
plt.scatter(x, my_df["left_ci"], color="blue", s=1)
plt.scatter(x, my_df["right_ci"], color="blue", s=1)
plt.show()
plt.clf()

