import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
import os


berry_train = pd.read_csv("./data/blueberry/train.csv")
berry_test = pd.read_csv("./data/blueberry/test.csv")
sub_df = pd.read_csv("./data/blueberry/sample_submission.csv")

berry_train.isna().sum()
berry_test.isna().sum()

x = berry_train.drop(["id", "yield"], axis = 1)
y = berry_train["yield"]


### Lasso
kf = KFold(n_splits=15, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, x, y, cv = kf,
                                     n_jobs = -1,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0.020, 0.025, 0.0001)
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
test_X = berry_test.drop("id", axis=1)

test_X.isna().sum()

model = Lasso(alpha = 0.006769999999999969)

# 모델 학습
model.fit(x, y)

pred_y=model.predict(test_X)

# SalePrice 바꿔치기
sub_df["yield"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("data/blueberry/sample_submission3.csv", index=False)

### Ridge (알파 = 0.02499)
kf = KFold(n_splits=20, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, x, y, 
                                     cv = kf,
                                     n_jobs = -1,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0.020, 0.025, 0.0001)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    ridge = ridge(alpha=alpha)
    mean_scores[k] = rmse(ridge)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Ridge Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

#==========================================
model = Ridge(alpha = 0.006769999999999969)

# 모델 학습
model.fit(x, y)

pred_y=model.predict(test_X)

# SalePrice 바꿔치기
sub_df["yield"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("data/blueberry/sample_submission3.csv", index=False)

#### 선형회귀 직선
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 성능 측정 ()
pred_y=model.predict(test_X)

# yield 바꿔치기
sub_df["yield"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("./data/blueberry/sample_submission2.csv", index=False)

###############################################################
# KNN 활용
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer, mean_squared_error


x = berry_train.drop(["id","yield"], axis=1)
y = berry_train["yield"]

test_X = berry_test.drop("id", axis=1)

# Validation(교차검증) 만들기(10번)
kf = KFold(n_splits=2, shuffle=True, random_state=2024)

# valid_result 한 칸 계산하는 함수
def rmse(model):
    # -- -> + 로 만들어준 것
    score = np.sqrt(-cross_val_score(model, 
                                     x, 
                                     y, 
                                     cv = kf,
                                     n_jobs=-1, 
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

# KNN 모델에서 사용할 k 값을 테스트하기 위한 범위 설정
k_values = np.arange(14, 25)
# 막대기 만들어줌(valid_result)
mean_scores = np.zeros(len(k_values))

k_index = 0
# for문으로 칸 다 채워서 막대기 채움
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    mean_scores[k_index] = rmse(knn)
    k_index += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'k': k_values,
    'validation_error': mean_scores
})

df

# 결과 시각화
plt.plot(df['k'], df['validation_error'], label='Validation Error', color='blue')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('KNN Regression Train vs Validation Error')
plt.show()

# 최적의 k 값 찾기
optimal_k = df['k'][np.argmin(df['validation_error'])]
print("Optimal k:", optimal_k)

# 최적의 k를 사용한 KNN 모델 생성
model = KNeighborsRegressor(n_neighbors=int(optimal_k))

# 모델 학습
model.fit(x, y)

# 예측
pred_y = model.predict(test_X)

# 결과를 submission 파일로 저장
sub_df["yield"] = pred_y
sub_df.to_csv("./data/blueberry/sample_submission_knn.csv", index=False)






####################################################################
# 가중치 부여하기 
ridge = pd.read_csv("./data/blueberry/ridge.csv")
lasso = pd.read_csv("./data/blueberry/lasso.csv")
linear = pd.read_csv("./data/blueberry/linear.csv")
knn = pd.read_csv("./data/blueberry/sample_submission_knn.csv")


yield_ri = ridge["yield"] # 예진
yield_la = lasso["yield"] # 서연
yield_li = linear["yield"] # 유나
yield_kn = knn["yield"] 


yield_total = ((yield_ri * 6) + (yield_la * 4))/10

sub_df["yield"] = yield_total

# csv 파일로 내보내기
sub_df.to_csv("./data/blueberry/submission_total3.csv", index=False)