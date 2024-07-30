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
