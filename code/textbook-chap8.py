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
