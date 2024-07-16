import pandas as pd
import numpy as np

exam = pd.read_csv("data/exam.csv")

exam["nclass"]
exam[["nclass"]]
exam[["id", "nclass"]]
exam.drop(columns = ["math", "english"])
exam


exam.query("nclass == 1")[["math", "english"]]
exam.query("nclass == 1") \
    [["math", "english"]] \
    .head()

# 정렬하기
exam.sort_values("math")
exam.sort_values("math", ascending = False)
exam.sort_values(["nclass", "english"], ascending = [True, False])

#변수추가
exam = exam.assign(
        total = exam["math"] + exam["english"] + exam["science"],
        mean = (exam["math"] + exam["english"] + exam["science"])/3
        )\
        .sort_values("total", ascending = False)
exam.head()        



# lambda 함수 사용하기
exam2 = pd.read_csv("data/exam.csv")
exam2 = exam2.assign(
        total = lambda x: x["math"] + x["english"] + x["science"],
        mean = lambda x: x["total"]/3
        ) \
        .sort_values("total", ascending = False)
exam2.head()        


# 그룹을 나눠 요약을 하는 .groupby() + .agg() 콤보
exam2.agg(mean_math = ("math", "mean"))
exam2.groupby("nclass") \
     .agg(
         mean_math = ("math", "mean"),
         mean_eng = ("english", "mean"),
         mean_sci = ("science", "mean"),
    )
    
    
import pydataset

mpg = pd.read_csv("data/mpg.csv")

df = pydataset.data("mpg")     

mpg.query('category =="suv"') \
    .assign(total = (mpg["hwy"] + mpg["cty"]) / 2) \
    .groupby("manufacturer") \
    .agg(mean_tot = ( "total", "mean")) \
    .sort_values("mean_tot", ascending = False) \
    .head()
# 숙제 p144, p153, p158


# 1. 변수 이름 변경 했는지?
# 2. 행들을 필터링 했는지?
# 3. 새로운 변수를 생성했는지?
# 4. 그룹 변수 기준으로 요약을 했는지?
# 5. 정렬 했는지?
