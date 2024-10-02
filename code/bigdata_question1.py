import pandas as pd


#데이터 불러오기
train =  pd.read_csv('../data/mart_train.csv') # (700,10)
test =  pd.read_csv('../data/mart_test.csv') # (300,9)

train.head(5)

target = train.pop('total') 

print(train.shape, target.shape, test.shape)

train.info()

# 스케일러
cols = ['rating']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.fit_transform(test[cols])

train.head()

# 더미 처리
print(train.shape, test.shape) #(700,9) , (300,9)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
print(train.shape, test.shape) #(700,30), (300,30)

# train valid 분할
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(train, target, test_size = 0.3, random_state=2024)
print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)
#(560,30) (140,30) (560,) (140,)

# 랜덤 포레스트
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
model = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_split=5, min_samples_leaf=20, max_features= 'log2')
model.fit(X_tr, y_tr)
pred = model.predict(X_val)
pred.shape # (140,)

param_grid={
    'max_depth': [3, 5, 7],
    'min_samples_split': [20, 10, 5],
    'min_samples_leaf': [5, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None]
}
grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(X_tr, y_tr)
mart_rf_model1=grid_search.best_estimator_
# max_depth=3, min_samples_split=5, min_samples_leaf=20, max_features= 'log2'


from sklearn.metrics import mean_squared_error
print(mean_squared_error(pred, y_val)**0.5)

pred = model.predict(test)

submit = pd.DataFrame({'pred':pred})
submit.to_csv('submit.csv', index=False)
submit = pd.read_csv('submit.csv')
submit