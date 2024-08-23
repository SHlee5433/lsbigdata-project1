# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)
## 필요한 데이터 불러오기
house_train=pd.read_csv("lsbigdata-project1/data/houseprice/train.csv")
house_test=pd.read_csv("lsbigdata-project1/data/houseprice/test.csv")
sub_df=pd.read_csv("lsbigdata-project1/data/houseprice/sample_submission.csv")

# 트레인, 테스트 합치기(더미변수 만드는거 한 번에 처리하기 위해서 더하는거.)
combine_df = pd.concat([house_train, house_test], ignore_index = True) # ignore_index 옵션이 있음.

# 더미변수 만들기
neighborhood_dummies = pd.get_dummies(
    combine_df["Neighborhood"],
    drop_first=True
)

# 더미데이터를 train, test로 데이터 나누기
train_dummies = neighborhood_dummies.iloc[:1460,]

test_dummies = neighborhood_dummies.iloc[1460:,]
test_dummies = test_dummies.reset_index(drop=True) # 인덱스를 초기화(house_test 원본 데이터와 맞춰야) 잘 합쳐짐

# 원래 데이터에서 필요한 변수들만 골라서 더미데이터를 합치기.
my_train = pd.concat([house_train[["SalePrice", "GrLivArea", "GarageArea"]],
               train_dummies], axis=1)

my_test_x = pd.concat([house_test[["GrLivArea", "GarageArea"]],
               test_dummies], axis=1)

# train 데이터의 길이 구하기
train_n = len(my_train) # 1460

## Validation 셋(모의고사 셋) 만들기
np.random.seed(42)
val_index = np.random.choice(np.arange(train_n), size = 438,
                 replace = False) #30% 정도의 갯수를 랜덤으로 고르기.
val_index

new_valid = my_train.loc[val_index]  # 30% 438개
new_train = my_train.drop(val_index) # 70% 1022개

######## 이상치 탐색 및 없애기
new_train = new_train.query("GrLivArea <= 4500") # 나중에 실행하지 말고도 구해보기.

# train 데이터의 길이 구하기
len(new_train) # 1020

# train 데이터 가격 분리하기.
train_x = new_train.iloc[:,1:]
train_y = new_train[["SalePrice"]]

valid_x = new_valid.iloc[:,1:]
valid_y = new_valid[["SalePrice"]]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)

# 성능 측정
y_hat = model.predict(valid_x)
np.mean(np.sqrt((valid_y-y_hat)**2)) #26265

# 위에서 이상치 없애기를 하지 않았다면?
# 25820
# 더 낮은데?