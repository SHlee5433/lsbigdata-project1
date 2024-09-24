# GBRT

import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X = np.random.rand(100) - 0.5
y = 3 * X**2 + 0.05*np.random.randn(100)
X=X.reshape(100,1)

import matplotlib.pyplot as plt
plt.scatter(x = X, y = y)

tree_model1=DecisionTreeRegressor(max_depth=2)
tree_model1.fit(X, y)
y_tree1 = tree_model1.predict(X)

# 1차 트리 예측값 시각화
plt.scatter(x = X, y = y)
plt.scatter(x= X, y = y_tree1)

y2=y-tree_model1.predict(X)
tree_model2=DecisionTreeRegressor(max_depth=2)
tree_model2.fit(X, y2) # 여기서 y는 잔차
y_tree2 = tree_model2.predict(X)

# 2번째 y 데이터(y-1차 트리 예측값)
plt.scatter(x = X, y = y2)
plt.scatter(x = X, y = y_tree2)

# 1차 + 2차 트리 예측값 시각화
plt.scatter(x = X, y = y)
plt.scatter(x= X, y = y_tree1 + y_tree2)

y3=y2-tree_model2.predict(X)
tree_model3=DecisionTreeRegressor(max_depth=2)
tree_model3.fit(X, y3)
y_tree3 = tree_model3.predict(X)

# 세번째 y 데이터 (y2-2차 트리 예측값)
plt.scatter(x = X, y = y3)
plt.scatter(x = X, y = y_tree3)

# 1차 + 2차 + 3차 트리 예측값 시각화
l_rate = 1
plt.scatter(x = X, y = y)
plt.scatter(x= X, y = y_tree1 + l_rate * y_tree2 + l_rate * y_tree3)

# 새로운 데이터
X_new=np.array([[0.5]])

tree_model1.predict(X_new) + tree_model2.predict(X_new) + tree_model3.predict(X_new)

# 위 내용을 scikit-learn을 사용해서 구현
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth = 2,
                                 n_estimators = 3,
                                 learning_rate = 1.0,
                                 random_state = 42)

gbrt.fit(X, y)
gbrt.predict(X)