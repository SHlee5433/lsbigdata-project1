import pandas as pd
import statsmodels.api as sm
# Q1.
# 데이터를 로드하고, 로지스틱 회귀모델을 적합하고, 회귀 표를 작성하세요

# 데이터 로드하기
df = pd.read_table('../data/leukemia_remission.txt',delim_whitespace=True)

df

# 로지스틱 회귀모델 적합 후, 화귀 표

model = sm.formula.logit("REMISS ~ CELL + SMEAR + INFIL + LI + BLAST + TEMP", data = df).fit()

print(model.summary())

# Q2.
# 해당 모델은 통계적으로 유의한가요? 그 이유를 검정통계량를 사용해서 설명하시오.

from scipy.stats import chi2
# 검정통계량 : −2(ℓ(𝛽)̂ (0) − ℓ(𝛽)̂ )  =  -2*(-17.186+10.797)  = 12.779
1 - chi2.cdf(12.779, df=6)  # 0.0467

# 결론 : LLR p-value: 0.0467 < 유의수준 0.05보다 작으니까 통계적으로 유의하다고 할 수 있다.


# Q3.
# 유의수준 0.2를 기준으로 통계적으로 유의한 변수
# p-value가 0.2보다 작은 변수 = LI, TEMP

# Q4.
# 다음 환자에 대한 오즈는 얼마인가요?
# CELL (골수의 세포성): 65%
# SMEAR (골수편의 백혈구 비율): 45%
# INFIL (골수의 백혈병 세포 침투 비율): 55%
# LI (골수 백혈병 세포의 라벨링 인덱스): 1.2
# BLAST (말초혈액의 백혈병 세포 수): 1.1세포/μL
# TEMP (치료 시작 전 최고 체온): 0.9

odds = np.exp(64.258 + 30.830 * 0.65 + 24.686 * 0.45 + (-24.975) * 0.55 + 4.361 * 1.2 + (-0.012) * 1.1 + (-100.173) * 0.9)

odds  # 0.038

# Q5. 위 환자의 혈액에서 백혈병 세포가 관측되지 않은 확률은 얼마인가요?

odds / (1 + odds) # 0.037

# Q6. TEMP 변수의 계수는 얼마이며, 해당 계수를 사용해서 TEMP 변수가 백혈병 치료에 대한 영향을 설명하시오.

# TEMP 변수의 계수 = -100.173
# 오즈비 3.127
# TEMP가 1 올라가면 로그 오즈는 157.86만큼 감소하며, 백혈병 상태에 도달할 가능성이 크게 줄어든다

np.exp(-100.173)

# Q7. CELL 변수의 99% 오즈비에 대한 신뢰구간을 구하시오.
cell_beta = 30.830
z = 2.58
std_err = 52.135

upper = cell_beta + z * std_err
lower = cell_beta - z * std_err

# CELL beta의 신뢰구간
upper # 165.34
lower # -103.68

# CELL beta 오즈비의 신뢰구간
np.exp(upper) # 6.39
np.exp(lower) # 9.40

# Q8. 주어진 데이터에 대하여 로지스틱 회귀 모델의 예측 확률을 구한 후, 50% 이상인 경우 1로 처리하여, 혼동 행렬를 구하시오.

# from sklearn.metrics import confusion_matrix
# 
# pred_remiss = model.predict(df) # 예측 확률
# 
# pred_remiss = pd.DataFrame(pred_remiss, columns=['pred_remiss'])
# pred_df = np.where(pred_remiss > 0.5, "1", "0")
# 
# actual_classes = df['REMISS']
# conf_matrix = confusion_matrix(actual_classes, pred_df)


from sklearn.metrics import confusion_matrix

# 1. 모델을 사용하여 예측 확률을 계산
pred_probs = model.predict(df)

# 2. 50% 기준으로 이진화 (0 또는 1로 변환)
predictions = [1 if prob > 0.5 else 0 for prob in pred_probs]

# 3. 실제 값 (df['REMISS'])과 예측 값 (predictions) 비교하여 혼동 행렬 계산
conf_matrix = confusion_matrix(df['REMISS'], predictions)

# 혼동 행렬 출력
conf_matrix


# Q9. 해당 모델의 Accuracy는 얼마인가요?

# Accuracy = 전체 예측에서 옳은 예측의 비율

Accuracy = (15 + 5) / (15 + 3 + 4 + 5)
Accuracy # 0.741

# Q10. 해당 모델의 F1 Score를 구하세요.

pre = 5 / (5 + 3)
re = 5 / (5 + 4)

F1 = 2 * (pre * re) / (pre + re)
F1 # 0.588

