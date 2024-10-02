import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier



# 데이터 불러오기
raw_data = pd.read_csv("../data/bigdata/1주_실습데이터.csv")
data = raw_data.copy()

## 전처리

# 1. 데이터 확인
data.info() # 결측치 확인
data.describe()
data.head(10)

# Y 변수 (타겟 변수) 분포 확인
target_dist = data['Y'].value_counts()

# 상관관계 계산
correlation_matrix = data.corr()

# 상관관계 표 출력
print(correlation_matrix)

# 상관관계 행렬을 불러온 데이터프레임에서 계산
correlation_matrix = data.corr()

# 유의미한 상관관계 (상관계수 절댓값이 0.5 이상인 변수 쌍 찾기)
threshold = 0.5
significant_pairs = correlation_matrix[(correlation_matrix.abs() >= threshold) & (correlation_matrix != 1)]

# 유의미한 변수 쌍 목록 생성
significant_columns = significant_pairs.stack().index.tolist()

# 중복 제거 (예: (X1, Y)와 (Y, X1) 중복)
unique_pairs = set()
for pair in significant_columns:
    unique_pairs.add(tuple(sorted(pair)))

# 산점도 그리기
plt.figure(figsize=(15, 10))
for i, (var1, var2) in enumerate(unique_pairs):
    plt.subplot(3, 3, i + 1)  # 3x3 그리드에 산점도 배치
    sns.scatterplot(data=data, x=var1, y=var2, hue='Y', palette='deep')
    plt.title(f'Scatter Plot of {var1} vs {var2}')
    plt.xlabel(var1)
    plt.ylabel(var2)

plt.tight_layout()
plt.show()

# 2. 히트맵 시각화
plt.figure(figsize=(12, 8))  # 히트맵 크기 설정
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True, linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()

data.nunique() # -> x4랑 x13이 값이 하나임을 확인

data['X13']
data['X4']
# -> 둘 다 제거

data = data.drop(columns=['X4', 'X13'])

# 동일한 값을 가진 열 쌍을 찾는 코드
identical_columns = []
columns = data.columns

# 각 열을 비교하여 동일한 값을 가진 열을 찾기
for i in range(len(columns)):
    for j in range(i+1, len(columns)):
        if data[columns[i]].equals(data[columns[j]]):
            identical_columns.append((columns[i], columns[j]))

# 결과 출력
identical_columns
# [('X6', 'X20'), ('X8', 'X18'), ('X12', 'X19')]
# 이런 경우에는 drop으로 둘 중에 하나만 제거 (x8, x6, x12 남기기)
data = data.drop(columns=['X6', 'X8', 'X12'])


# IQR 방식으로 이상치 탐지
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# 이상치 탐지
outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
print("각 변수에서 탐지된 이상치 개수:")
print(outliers)

# 박스플롯 그리기 - 직사각형 배치
num_cols = len(data.columns)
nrows = (num_cols // 5) + (num_cols % 5 > 0)  # 3열로 나눠서 배치
fig, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(15, 3*nrows))

# 데이터가 3개 미만인 경우를 대비
axes = axes.flatten()

# 각 변수에 대해 박스플롯 그리기
for i, col in enumerate(data.columns):
    sns.boxplot(data[col], ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')

# 빈 플롯 숨기기 (컬럼 수가 3의 배수가 아닐 때)
for i in range(len(data.columns), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# 2. 이상치 제거: 상/하위 0%~1%, 0.1% 간격으로 제거하는 함수 정의
# IQR 방식으로 이상치 탐지하는 함수
def detect_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
    return outliers

# 이상치 제거 함수 (단계별 상하위 백분위수 기준 제거)
def remove_outliers_by_percentile(data, lower_percentile, upper_percentile):
    lower_bounds = data.quantile(lower_percentile)
    upper_bounds = data.quantile(upper_percentile)

    # 각 열의 값이 해당 백분위수 기준 안에 있는지 필터링
    cleaned_data = data.apply(lambda x: x[(x >= lower_bounds[x.name]) & (x <= upper_bounds[x.name])], axis=0)

    return cleaned_data.dropna()

# 0%부터 1%까지 0.1%씩 단계적으로 이상치 제거 및 탐지
step = 0.001  # 0.1%
percentiles = np.arange(0, 0.011, step)  # 0% ~ 1%까지의 백분위수

# 결과를 저장할 데이터프레임 생성
outliers_df = pd.DataFrame()

for perc in percentiles:
    # 현재 백분위수로 데이터 이상치 제거
    clean_data = remove_outliers_by_percentile(data, perc, 1-perc)

    # 이상치 개수 탐지
    outliers = detect_outliers(clean_data)

    # 결과를 데이터프레임에 추가 (백분위수와 함께)
    outliers_df[f'{perc*100:.1f}% 제거'] = outliers

    def apply_scaling(clean_data, normalize=True):
        if normalize:
            scaler = RobustScaler()
            scaled_data = pd.DataFrame(scaler.fit_transform(clean_data), columns=clean_data.columns)
            return scaled_data
        return clean_data

def evaluate_model_normalize(clean_data, model_type='XGB',normalize = True):
    X = clean_data.drop(columns=['Y'])
    y = clean_data['Y']

    X = apply_scaling(X, normalize)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 모델 선택
    if model_type == 'XGB':
        model = XGBClassifier()
    elif model_type == 'LGBM':
        model = LGBMClassifier()
    elif model_type == 'RF':
        model = RandomForestClassifier()

    # 모델 학습
    model.fit(X_train, y_train)

    # 예측 및 성능 평가
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return auroc, f1, recall, precision

def evaluate_model(clean_data, model_type='XGB', nomarlize = True):
    X = clean_data.drop(columns=['Y'])
    y = clean_data['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # 타겟 변수 y의 클래스 비율을 학습 데이터와 테스트 데이터에 동일하게 유지

    # 모델 선택
    if model_type == 'XGB':
        model = XGBClassifier()
    elif model_type == 'LGBM':
        model = LGBMClassifier()
    elif model_type == 'RF':
        model = RandomForestClassifier()

    # 모델 학습
    model.fit(X_train, y_train)

    # 예측 및 성능 평가
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return auroc, f1, recall, precision

X = clean_data.drop(columns=['Y'])
y = clean_data['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clean_data

results = []
model_types = ['XGB']

for i in np.arange(0, 0.011, 0.001):
    clean_data = remove_outliers_by_percentile(data, i, 1-i)

    for normalize in [True, False]:
            if normalize == True:
                        for model_type in model_types:
                                    auroc, f1, recall, precision = evaluate_model_normalize(clean_data, model_type, normalize)
                                    results.append({
                                    'Outlier_Percentage': i,
                                    'Normalized': normalize,
                                    'Model_Type': model_type,
                                    'AUROC': auroc,
                                    'F1-Score': f1,
                                    'Recall': recall,
                                    'Precision': precision
                                    })
            else:
                        for model_type in model_types:
                                    auroc, f1, recall, precision = evaluate_model(clean_data, model_type)
                                    results.append({
                                    'Outlier_Percentage': i,
                                    'Normalized': normalize,
                                    'Model_Type': model_type,
                                    'AUROC': auroc,
                                    'F1-Score': f1,
                                    'Recall': recall,
                                    'Precision': precision
                                    })


# 결과 출력 (테이블 형태로)
results_df = pd.DataFrame(results)
print(results_df)

## SHAP 분석을 통한 모델 해석
# 이상치 제거 및  XGBoost 모델 학습
import shap

results = []
model_types = ['XGB']

for i in np.arange(0, 0.011, 0.001):
    clean_data = remove_outliers_by_percentile(data, i, 1-i)

    for normalize in [True, False]:
        for model_type in model_types:
            auroc, f1, recall, precision = evaluate_model(clean_data, model_type, normalize)
            results.append({
                'Outlier_Percentage': i,
                'Normalized': normalize,
                'Model_Type': model_type,
                'AUROC': auroc,
                'F1-Score': f1,
                'Recall': recall,
                'Precision': precision
            })

# 결과 출력 (테이블 형태로)
results_df = pd.DataFrame(results)
print(results_df)

# XGBoost 모델 정의 및 학습
X = clean_data.drop(columns=['Y'])  # 'Y'는 타겟 변수
y = clean_data['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier()  # XGBoost 모델 정의
model.fit(X_train, y_train)  # 모델 학습

# SHAP 분석
explainer = shap.TreeExplainer(model)  # Tree 모델용 Explainer
shap_values = explainer.shap_values(X_test)

# 1-1 전체 변수의 중요도 시각화(Summary Plot)
shap.summary_plot(shap_values, X_test)

# 1-2 각 변수의 SHAP값 분포 시각화(Bar Plot)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# 1-3  특정 데이터 샘플에 대한 Force Plot (개별 예측 설명)
sample_idx = 10  # 예시로 10번째 샘플에 대한 예측 설명
shap.force_plot(explainer.expected_value, shap_values[sample_idx], X_test.iloc[sample_idx, :], matplotlib=True)

# 1-4 특정 변수에 대한 Partial Dependence Plot (PDP)
# 예시로 'X1' 변수에 대한 SHAP 분석
shap.dependence_plot('X1', shap_values, X_test)

# 2. 의사결정나무
# 원본 데이터 나누기
data = raw_data.copy()
X = data.drop(columns=['Y'])
y = data['Y']

data.shape
X.shape
X_train

# 의사결정나무 모델 학습 (최대 깊이를 제한해서 트리가 너무 복잡해지지 않도록 함)
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X, y)

# 의사결정나무 시각화 (큰 이미지 크기로 설정)
plt.figure(figsize=(15, 10))  # 이미지 크기를 크게 설정
plot_tree(tree_model, feature_names=X.columns, class_names=['Class 0', 'Class 1'], filled=True, rounded=True)
plt.title("Decision Tree Visualization with max_depth=3")
plt.show()

# X3이 0.406 이하일 때의 Y 값 분포 확인
below_threshold = data[data['X3'] <= 0.406]['Y'].value_counts()

# X3이 0.406보다 클 때의 Y 값 분포 확인
above_threshold = data[data['X3'] > 0.406]['Y'].value_counts()

# 결과 출력
print("X3 <= 0.406일 때 Y 값 분포:")
print(below_threshold)

print("\nX3 > 0.406일 때 Y 값 분포:")
print(above_threshold)