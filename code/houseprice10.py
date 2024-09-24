import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
import os


house_train = pd.read_csv("./data/houseprice/train.csv")
house_test = pd.read_csv("./data/houseprice/test.csv")
sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")

## NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
house_train.isna().sum()
house_test.isna().sum()

## 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    mode_value = house_train[col].mode()[0]  # 최빈값 계산
    house_train[col].fillna(mode_value, inplace=True)  # 최빈값으로 채우기

### house_test 결측치 채우기

## 숫자형 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_test[col].fillna(house_test[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_test.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    mode_value = house_test[col].mode()[0]  # 최빈값 계산
    house_test[col].fillna(mode_value, inplace=True)  # 최빈값으로 채우기

train_n=len(house_train)
# 최적의 람다값 구해서, 람다를 사용해서 houseprice test셋에 대한 예측

# 통합 df 만들기 + 더미코딩
# house_test.select_dtypes(include=[int, float])

df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns = df.select_dtypes(include=[object]).columns,
    drop_first = True
    )
df

# train / test 데이터셋
train_df = df.iloc[:train_n,]
test_df = df.iloc[train_n:,]

x = train_df.drop("SalePrice", axis=1)
x = x.drop("Id", axis = 1)
y = train_df["SalePrice"]

# 교차 검증 설정
kf = KFold(n_splits=10, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, x, y, cv = kf,
                                     n_jobs = -1,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(147, 149, 0.01)
mean_scores = np.zeros(len(alpha_values))

k=0

for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# test_df 만들기
test_X = test_df.drop(["Id", "SalePrice"], axis=1)

test_X.isna().sum()



#==========================================
# 최적의 alpha 값 148.04
# 선형 회귀 모델 생성(8번)
model = Lasso(alpha=148.04)

# 모델 학습
model.fit(x, y)


pred_y=model.predict(test_X)

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("data/houseprice/sample_submission10.csv", index=False)


