import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.metrics import roc_auc_score, precision_score,recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler


##Standard Scaler
#Scaler_std = StandardScaler()
#train_x_std_scaled = Scaler_std.fit_transform(X_train)
#test_x_std_scaled = Scaler_std.transform(X_test)


##Min_Max Scaler
#Scaler_mm = MinMaxScaler()
#train_x_mm_scaled = Scaler_mm.fit_transform(X_train)
#test_x_mm_scaled = Scaler_mm.transform(X_test)



sns.set_style('whitegrid')  

df_raw = pd.read_csv("../data/bigdata/data_week2.csv", encoding ="cp949")

df = df_raw.copy()

df.columns


df.columns = ['num', 'datetime', 'target', 'temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df["datetime"] = pd.to_datetime(df["datetime"])
df['month'] = df.datetime.dt.month
df['day'] = df.datetime.dt.day
df['hour'] = df.datetime.dt.hour
df['weekday'] = df.datetime.dt.weekday
df['dayofyear'] = df.datetime.dt.dayofyear
eda_df = df.copy()
df["target"].describe()
df["target"].hist()

df["temp"].describe()
df["temp"].hist()

df["wind"].describe()
df["wind"].hist()

df["humid"].describe()
df["humid"].hist()

df["rain"].describe()
df["rain"].hist()

df.drop("datetime",axis=1,inplace=True)


df["target"] = np.log(df["target"]+1)

by_weekday = df.groupby(['num','weekday'])['target'].median().reset_index().pivot(index='num', columns='weekday', values='target').reset_index()
by_hour = df.groupby(['num','hour'])['target'].median().reset_index().pivot(index='num', columns='hour', values='target').reset_index().drop('num', axis=1)
clus_df = pd.concat([by_weekday, by_hour], axis= 1)
columns = ['num'] + ['day'+str(i) for i in range(7)] + ['hour'+str(i) for i in range(24)]
clus_df.columns = columns

for i in range(len(clus_df)):
    # 요일 별 전력 중앙값에 대해 scaling
    clus_df.iloc[i,1:8] = (clus_df.iloc[i,1:8] - clus_df.iloc[i,1:8].mean())/clus_df.iloc[i,1:8].std()
    # 시간대별 전력 중앙값에 대해 scaling
    clus_df.iloc[i,8:] = (clus_df.iloc[i,8:] - clus_df.iloc[i,8:].mean())/clus_df.iloc[i,8:].std()



# 클러스터링
def change_n_clusters(n_clusters, data):
    sum_of_squared_distance = []
    for n_cluster in n_clusters:
        kmeans = KMeans(n_clusters=n_cluster)
        kmeans.fit(data)
        sum_of_squared_distance.append(kmeans.inertia_)
    
change_n_clusters([2,3,4,5,6,7,8,9,10,11], clus_df.iloc[:,1:])

kmeans = KMeans(n_clusters=4, random_state = 42)
km_cluster = kmeans.fit_predict(clus_df.iloc[:,1:])

df['km_cluster'] = km_cluster.repeat(122400/60)



train = []
valid = []
for num, group in df.groupby('num'):
    train.append(group.iloc[:len(group)-7*24])  
    valid.append(group.iloc[len(group)-7*24:]) 


train_df = pd.concat(train)
train_x = train_df.drop("target",axis=1)
train_y = train_df["target"] 

valid_df = pd.concat(valid)
valid_x = valid_df.drop("target",axis=1)
valid_y = valid_df["target"] 


# cl0_df = df[df["km_cluster"] == 0] # 38362개
# X0 = cl0_df.drop(columns=['target'])
# y0 = cl0_df['target']

# X0_train, X0_valid, y0_train, y0_valid = temporal_train_test_split(X0, y0, test_size = 168) 

# cl1_df = df[df["km_cluster"] == 1] # 70177개
# X1 = cl1_df.drop(columns=['target'])
# y1 = cl1_df['target']

# X1_train, X1_valid, y1_train, y1_valid = temporal_train_test_split(X1, y1, test_size = 168) 

# cl2_df = df[df["km_cluster"] == 2] # 4913개
# X2 = cl2_df.drop(columns=['target'])
# y2 = cl2_df['target']

# X2_train, X2_valid, y2_train, y2_valid = temporal_train_test_split(X2, y2, test_size = 168) 

# cl3_df = df[df["km_cluster"] == 3] # 8948개 
# X3 = cl3_df.drop(columns=['target'])
# y3 = cl3_df['target']

# X3_train, X3_valid, y3_train, y3_valid = temporal_train_test_split(X3, y3, test_size = 168) 




##### xgboost
def xgb(train_x,train_y,valid_x,valid_y):
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(train_x, train_y)

    xgb_pred = xgb_model.predict(valid_x)
    xgb_rmse = np.sqrt(mean_squared_error(valid_y, xgb_pred))
    return xgb_rmse 


#####lgbm
def lgbm(train_x, train_y, valid_x, valid_y):
    lgb_model = LGBMRegressor(random_state=42)
    lgb_model.fit(train_x, train_y)

    lgb_pred = lgb_model.predict(valid_x)
    lgb_rmse = np.sqrt(mean_squared_error(valid_y, lgb_pred))
    return lgb_rmse 


#####catboost
def cat(train_x, train_y, valid_x, valid_y):
    cat_model = CatBoostRegressor(random_state=42, verbose=0)  # verbose=0: 학습 로그 출력하지 않음
    cat_model.fit(train_x, train_y)

    cat_pred = cat_model.predict(valid_x)
    cat_rmse = np.sqrt(mean_squared_error(valid_y, cat_pred))
    return cat_rmse 


result = []
for x in range(4):
    # 클러스터별 훈련 데이터 가져오기
    df_tmp = df.loc[df["km_cluster"] == x, :]
    train_x_tmp = train_x.loc[train_x["km_cluster"] == x, :]
    train_y_tmp = train_y.loc[train_x_tmp.index]
    
    # 클러스터별 검증 데이터 가져오기
    valid_x_tmp = valid_x.loc[valid_x["km_cluster"] == x, :]
    valid_y_tmp = valid_y.loc[valid_x_tmp.index]

    # 데이터가 있는지 확인

    weight = len(df_tmp) / len(df)
    
    # 모델 학습 및 평가
    xgb_rmse = xgb(train_x_tmp, train_y_tmp, valid_x_tmp, valid_y_tmp)
    lgbm_rmse = lgbm(train_x_tmp, train_y_tmp, valid_x_tmp, valid_y_tmp)
    cat_rmse = cat(train_x_tmp, train_y_tmp, valid_x_tmp, valid_y_tmp)
    
    # 결과 저장
    result.append({
        "cluster": x,
        "xgb_rmse": xgb_rmse,
        "lgbm_rmse": lgbm_rmse,
        "catboost_rmse": cat_rmse,
        "weight": weight
    })
    

fig = plt.figure(figsize=(20, 4))
for c in range(4):
    temp = df[df.km_cluster == c]
    # weekday와 hour 별로 그룹화하여 target의 중앙값 계산
    temp = temp.groupby(['weekday', 'hour'])['target'].median().reset_index().pivot(index='weekday', columns='hour', values='target')
    
    plt.subplot(1, 4, c+1)  # 4개의 클러스터를 표현하기 위한 subplot
    sns.heatmap(temp, cmap="coolwarm", cbar_kws={'label': 'Median Target'})  # 컬러맵과 컬러바 설정
    plt.title(f'Cluster {c}')
    plt.xlabel('Hour')  # X축을 시간으로 설정
    plt.ylabel('Weekday')  # Y축을 요일로 설정
    
result



# 건물별로 모델 돌리기?


# num_df=[]
# 
# for i in range(1, 61):
#     num_df.append(df[df["num"] == i])
# 
# num_df


from sklearn.metrics import mean_squared_error
import numpy as np

# 건물별 xgboost rmse값 구하기
# RMSE를 계산하는 함수 정의
def calculate_rmse_for_num(df, num):
    # num 값에 따라 데이터를 필터링
    num_df = df[df['num'] == num]
    
    # 데이터를 feature와 target으로 분리
    X = num_df.drop(columns=['target'])
    y = num_df['target']
    
    # 데이터를 훈련 셋과 검증 셋으로 나누기
    train_size = len(X) - 168  # 검증 셋을 가장 최근 168개로 설정 (예: 24x7)
    X_train, X_valid = X[:train_size], X[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]
    
    # 간단한 모델 예시 (XGBoost, LGBM, CatBoost 사용 가능)
    model = XGBRegressor(random_state=42)
    
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_valid)
    
    # RMSE 계산
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    
    return rmse


xgb_rmse_results = []  # 결과를 저장할 리스트

# 1부터 60까지의 num 값에 대해 RMSE 계산
for num in range(1, 61):
    rmse = calculate_rmse_for_num(df, num)  # 각 num에 대한 rmse 계산
    xgb_rmse_results.append({'num': num, 'rmse': rmse})  # num과 rmse 값을 저장

# 결과 출력
for result in xgb_rmse_results:
    print(f"num: {result['num']}, RMSE: {result['rmse']}")


# 건물 별 lgbm rmse 구하가

def calculate_lgbm_for_num(df, num):
    # num값에 따라 데이터를 필터링
    num_df = df[df["num"] == num]

    X = num_df.drop(columns = ["target"])
    y = num_df["target"]

    train_size = len(X) - 168

    X_train, X_valid = X[:train_size], X[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]

    model = LGBMRegressor(random_state = 42)
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    y_pred.hist()
    y.hist()

    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

    return rmse

lgbm_rmse_results = []
for num in range(1, 61):
    rmse = calculate_lgbm_for_num(df, num)
    lgbm_rmse_results.append({'num' : num, 'rmse' : rmse})

for result in lgbm_rmse_results :
    print(f"num: {result['num']}, RMSE: {result['rmse']}")


# 건물 별 catboost rmse 값 구하기

def calculate_cat_for_num(df, num):
    # num값에 따라 데이터를 필터링
    num_df = df[df["num"] == num]

    X = num_df.drop(columns = ["target"])
    y = num_df["target"]

    train_size = len(X) - 168

    X_train, X_valid = X[:train_size], X[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]

    model = CatBoostRegressor(random_state = 42)
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

    return rmse

cat_rmse_results = []
for num in range(1, 61):
    rmse = calculate_cat_for_num(df, num)
    cat_rmse_results.append({'num' : num, 'rmse' : rmse})

for result in cat_rmse_results :
    print(f"num: {result['num']}, RMSE: {result['rmse']}")



xgb = pd.DataFrame(xgb_rmse_results)

xgb_rmse_mean = xgb["rmse"].mean() # 0.210

lgbm = pd.DataFrame(lgbm_rmse_results)  

lgbm_rmse_mean = lgbm["rmse"].mean() # 0.189

cat = pd.DataFrame(cat_rmse_results)

cat_rmse_mean = cat["rmse"].mean() # 0.184
