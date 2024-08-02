house_train = pd.read_csv("./data/houseprice/train.csv")
house_test = pd.read_csv("./data/houseprice/test.csv")
sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")

# 이상치 탐색
house_train = house_train.query("GrLivArea <= 4500")

# x = np.array(house_train[["GrLivArea", "GarageArea"]]).reshape(-1, 2) # reshape 하기 위해선 꼭 np.array() 실시
# x = house_train["GrLivArea"] = 판다스 시리즈 (위에 칼럼 이름이 없음)
# x = house_train[["GrLivArea"]] = 판다스 프레임 (칼럼이 생기고 밑에 사이즈 표시)
# x = house_train["GrLivArea"] = 시리즈 (위에 칼럼 이름이 없음)
x = house_train[["GrLivArea", "GarageArea"]]
y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌 

# 시각화
# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

slope = model.coef_
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

def my_houseprice (x, y) :
    return model.coef_[0] * x + model.coef_[1] * y + model.intercept_

x = house_test["GrLivArea"]
y = house_test["GarageArea"]

my_houseprice(house_test["GrLivArea"], house_test["GarageArea"])




test_x = house_test[["GrLivArea", "GarageArea"]]
test_x

# 결측치 확인
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x = test_x.fillna(house_test["GarageArea"].mean())


pred_y = model.predict(test_x) # test 셋에 대한 집값
pred_y


# 기울기와 절편 출력
slope_grlivarea = model.coef_[0]
slope_garagearea = model.coef_[1]
intercept = model.intercept_

print(f"GrLivArea의 기울기 (slope): {slope_grlivarea}")
print(f"GarageArea의 기울기 (slope): {slope_garagearea}")
print(f"절편 (intercept): {intercept}")


# SalePrice 바꿔치기 

sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission7.csv", index = False)
sub_df
