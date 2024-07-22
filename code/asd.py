import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = sns.load_dataset("titanic")
sns.countplot(data = df, x = "class", hue = "alive")
plt.clf()
plt.show()

df =pd.DataFrame({ "name"  : ["김지훈", "이유진", "박동현", "김민지"],
                 "english" : [90, 80, 60, 70],
                 "math"    : [50, 60, 100, 20]})
df                 

df["name"]

df = pd.DataFrame({"제품"  : ["사과", "딸기", "수박"],
                   "가격"  : [1800, 1500, 3000],
                   "판매량": [24, 38, 13]})
                
df["가격"].mean()
df["판매량"].mean()

df_exam = pd.read_csv("data/exam.csv")
df_exam


df_raw = pd.DataFrame({"var1" : [1, 2, 3],
                      "var2" : [2, 3, 2]})
df_new = df_raw.copy()                      
df_new

df_new = df_new.rename(columns = {"var2" : "v2"})
df_new

df = pd.DataFrame({"var1" : [4, 3, 8],
                   "var2" : [2, 6, 1]})
df                   

df["var_sum"] = df["var1"] + df["var2"]
df["var_mean"] = (df["var1"] + df["var2"]) / 2
df

mpg = pd.read_csv("data/mpg.csv")
mpg

mpg["total"] = (mpg["cty"] + mpg["hwy"]) / 2 
mpg
mpg["total"].plot.hist()
plt.show()
plt.clf()


mpg["test"] = np.where(mpg["total"] > 20, "pass", "fail")
mpg

count_test = mpg["test"].value_counts()
count_test

count_test.plot.bar(rot = 0)
plt.show()
plt.clf()


mpg["grade"] = np.where(mpg["total"] >= 30, "A",
               np.where(mpg["total"] >= 20, "B", "C"))
mpg               

count_grade = mpg["grade"].value_counts()               
count_grade.plot.bar(rot = 0)

count_grade = count_grade.sort_index()

exam = pd.read_csv("data/exam.csv")
exam


exam["math"]    # 시리즈
exam["english"]
exam["science"]
df = exam[["nclass", "math", "english"]].head() #왜 []하나일 땐 추출이 안돼? #데이터 프레임
df

df.drop(columns = "math")
df.drop(columns = ["math", "english"])


exam.query("nclass == 1")["english"]
exam_new = exam.query("math >= 50")[["id", "math"]]
type(exam_new)

exam.query("math >= 50") \
           [["id", "math"]] \
           .head()

mpg = pd.read_csv("data/mpg.csv")
mpg_new = mpg[["category", "cty"]]
mpg_new.head()

mpg_a = mpg_new.query("category == 'suv'")["cty"].mean()
mpg_b = mpg_new.query("category == 'compact'")["cty"].mean()

mpg_a = mpg.query("manufacturer == 'audi'")\
        .sort_values("hwy", ascending = False)\
        .head()
mpg_a



exam.assign(total = exam["math"] + exam["english"]+ exam["science"])

exam.assign(
	total = exam["math"] + exam["english"] + exam["science"],       # total 추가
	mean = (exam["math"] + exam["english"] + exam["science"]) / 3)

import numpy as np
exam.assign(test = np.where(exam["science"] >= 60, "pass", "fail"))	

exam.assign(total = exam["math"]+ exam["english"] + exam["science"])\
	.sort_values("total")

exam.assign(total = lambda x: x["math"] + x["english"] + x["science"],
			mean = lambda x: x["total"]/3)	

exam.agg(mean_math = ("math", "mean"))			
exam["math"].mean()

exam.groupby("nclass") \
    .agg(mean_math = ("math", "mean"))
exam

exam.groupby("nclass", as_index = False) \
    .agg(mean_math = ("math", "mean"))

mpg.groupby(["manufacturer", "drv"], as_index = False) \
   .agg(mean_cty = ("cty", "mean"))
   
mpg.query("manufacturer == 'audi'") \
   .groupby(["drv"], as_index = False) \
   .agg(n = ("drv", "count"))
