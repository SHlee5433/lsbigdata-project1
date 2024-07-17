import pandas as pd
import numpy as np

df = pd.DataFrame({"sex": ["M", "F", np.nan, "M", "F"],

                   "score": [5, 4, 3, 4, np.nan]})
df        


pd.isna(df)


# 결측치 제거하기
df.dropna()                          # 모든 변수 결측치 제거거
df.dropna(subset = "score")          # score 변수에서 결측치 제거
df.dropna(subset = ["score", "sex"]) # 여러 변수 결측치 제거법

exam = pd.read_csv("data/exam.csv")
exam

# 데이터 프레임 location을 사용한 인덱싱
# exam.loc[행 인덱스, 열 인덱스] (리스트)
# exam.iloc[행 인덱스, 열 인덱스] (숫자)
exam.iloc[0:2, 0:4]
# exam.loc[2,7,4], ["math"]] = np.nan
exam.loc[[2, 7, 4], ["math"]] = np. nan
exam. iloc[[2, 7, 4], 2] = np.nan

exam

# 수학점수 50점 이하인 학생들 점수 다 50으로 상향 조정!

exam.loc[exam["math"] <= 50, "math"] = 50
exam
# 영어 점수 90점 이상인 학생들 90으로 하향 조정!(iloc 사용)
# iloc 조회는 안됨
exam.loc[exam["english"] >= 90, "english"]

#iloc을 사용해서 조회하려면 무조건 숫자벡터가 들어가야 함.
exam.iloc[exam["english"] >= 90, 3] # 실행 안됨
exam.iloc[np.array(exam["english"] >= 90), 3] # 실행 됨
exam.ilocnp.where([exam["english"] >= 90, 3] = 90)[0], 3} # np.where도 튜플이라 [0] 사용해서 꺼내오면 됨
exam.iloc[exam["english"] >= 90].index, 3] # index 벡터도 작동

# math 점수 50점 이하 "_" 변경
exam.loc[exam["math"] <= 50, "math"] = "_"
exam

# "_" 결측치를 수학점수 평균 바꾸고 싶은 경우
exam["math"]
