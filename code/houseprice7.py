house_train = pd.read_csv("./data/houseprice/train.csv")
house_test = pd.read_csv("./data/houseprice/test.csv")
sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")

# 이상치 탐색
# house_train = house_train.query("GrLivArea <= 4500")

## 회귀분석 적합(fit)하기
# x = np.array(house_train[["GrLivArea", "GarageArea"]]).reshape(-1, 2) # reshape 하기 위해선 꼭 np.array() 실시
# x = house_train["GrLivArea"] = 판다스 시리즈 (위에 칼럼 이름이 없음)
# x = house_train[["GrLivArea"]] = 판다스 프레임 (칼럼이 생기고 밑에 사이즈 표시)
# x = house_train["GrLivArea"] = 시리즈 (위에 칼럼 이름이 없음)
house_train["Neighborhood"]
neighborhood_dommies = pd.get_dummies(
    house_train["Neighborhood"],
    drop_first = True
)
x = pd.concat([house_train[["GrLivArea", "GarageArea"]],
                neighborhood_dommies], axis = 1)
y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌 

# 시각화
# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

neighborhood_dommies_test = pd.get_dummies(
    house_test["Neighborhood"],
    drop_first = True
    )
test_x = pd.concat([house_test[["GrLivArea", "GarageArea"]],
                neighborhood_dommies_test], axis = 1)

test_x

# 결측치 확인
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x = test_x.fillna(house_test["GarageArea"].mean())


pred_y = model.predict(test_x) # test 셋에 대한 집값
pred_y


# SalePrice 바꿔치기 

sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index = False)
sub_df
