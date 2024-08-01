import pandas as pd
import numpy as np

house_train = pd.read_csv("./data/houseprice/train.csv")
house_train = house_train[["Id", "YearBuilt", "SalePrice"]]
house_train.info()

# 연도별 평균
house_mean = house_train.groupby("YearBuilt", as_index = False) \
                        .agg(mean_year = ("SalePrice", "mean"))
house_mean

house_test=pd.read_csv("./data/houseprice/test.csv")
house_test=house_test[["Id", "YearBuilt"]]
house_test

house_test = pd.merge(house_test, house_mean,
                      how = "left", on = "YearBuilt")
house_test = house_test.rename(columns = {"mean_year" : "SalePrice"})
house_test

house_test["SalePrice"].isna().sum()

# 비어있는 테스트 세트 집들 확인

house_test.loc[house_test["SalePrice"].isna()]

# 집값 채우기

house_mean = house_train["SalePrice"].mean()
house_test["SalePrice"] = house_test["SalePrice"].fillna(house_mean)

# sub 데이터 불러오기

sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df

# SalePrice 바꿔치기

sub_df["SalePrice"] = house_test["SalePrice"]
sub_df

sub_df.to_csv("./data/houseprice/sample_submission2.csv", index = False)
sub_df

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
house_train = pd.read_csv("./data/houseprice/train.csv")
house_train = house_train[["Id", "YearBuilt", "YearRemodAdd", "BedroomAbvGr", "SalePrice"]]
house_train.info()

house_mean = house_train.groupby(["YearRemodAdd", "YearBuilt", "BedroomAbvGr"], as_index = False) \
                        .agg(mean_year = ("SalePrice", "mean"))
house_mean


house_test=pd.read_csv("./data/houseprice/test.csv")
house_test=house_test[["YearRemodAdd", "YearBuilt", "BedroomAbvGr"]]
house_test


house_test = pd.merge(house_test, house_mean, how = "left", on = ["YearRemodAdd", "YearBuilt", "BedroomAbvGr"])
house_test = house_test.rename(columns = {"mean_year" : "SalePrice"})
house_test

house_test["SalePrice"].isna().sum()

# 비어있는 테스트 세트 집들 확인

house_test.loc[house_test["SalePrice"].isna()]

# 집값 채우기

house_mean = house_train["SalePrice"].mean()
house_test["SalePrice"] = house_test["SalePrice"].fillna(house_mean)

# sub 데이터 불러오기

sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df

# SalePrice 바꿔치기

sub_df["SalePrice"] = house_test["SalePrice"]
sub_df

sub_df.to_csv("./data/houseprice/sample_submission3.csv", index = False)
sub_df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
house_train = pd.read_csv("./data/houseprice/train.csv")

df = house_train.dropna(subset=["MoSold","SalePrice"])\
                    .groupby("MoSold", as_index = False)\
                    .agg(count = ("SalePrice","count"))\
                    .sort_values("MoSold", ascending = True)
df

sns.barplot(data=df, x="MoSold", y="count", hue="MoSold")
plt.xlabel("월(month)")
plt.ylabel("이사횟수(count)")
plt.show()
plt.clf() 


몇 월에 이사를 가장 많이 갈까까
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


house_train2 = house_train[["YearBuilt", "OverallCond"]]

house_cond = house_train2.groupby("OverallCond", as_index = False)\
                    .agg(count = ("YearBuilt", "count"))\
                    .sort_values("count", ascending = False)
sns.barplot(data = house_cond, x = "OverallCond", y = "count", hue = "OverallCond")
plt.show()
plt.clf()

house_train3 = house_train[["BldgType", "OverallCond"]]

house_bed = house_train3.groupby(["OverallCond", "BldgType"], as_index = False)\
                    .agg(count = ("BldgType", "count"))\
                    .sort_values("count", ascending = False)
sns.barplot(data = house_bed, x = "OverallCond", y = "count", hue = "BldgType")
plt.show()
plt.clf()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df4 = house_train[["SalePrice", "Neighborhood"]]

house_bed = house_train.groupby(["Neighborhood", "SalePrice"], as_index = False)\
            .agg(region_mean = ("SalePrice", "mean"))\
            .sort_values("region_mean", ascending = False)
        
house_bed


df5 = df4["Neighborhood"].unique()
            
sns.barplot(data = house_bed, x = "Neighborhood", y = "region_mean", hue = "Neighborhood")
plt.show()
plt.clf()
