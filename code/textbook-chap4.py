import numpy as np
import pandas as pd

df = pd.DataFrame({
    'name': ["김지훈", "이유진", "박동현", "김민지"],
    'english': [90, 80, 60, 70],
    'math': [50, 60, 100, 20]
})

df
df["name"]
df[["name"]]

type(df)
type(df["name"])
type(df[["name"]])


# 평균
sum(df["english"])/4

# df[("name", "english")]
df[["name", "english"]]

df["name"]

pd.show_versions()

# !pip install openpyxl
import pandas as pd

df_exam=pd.read_excel("data/excel_exam.xlsx")
df_exam

sum(df_exam["math"])   /20
sum(df_exam["english"])/20
sum(df_exam["science"])/20

len(df_exam)
df_exam.shape
df_exam.size


df_exam=pd.read_excel("data/excel_exam.xlsx", 
                      sheet_name="Sheet2")
df_exam                      


df_exam["total"] = df_exam["math"] + df_exam["english"] + df_exam["science"]
df_exam["mean"] = df_exam["total"] / 3
df_exam

df_exam[df_exam["math"] > 50] 
df_exam[(df_exam["math"] > 50) & (df_exam["english"] > 50)] 


mean_m=np.mean(df_exam["math"])
mean_e=np.mean(df_exam["english"])

df_exam[(df_exam["math"] > mean_m) & 
                (df_exam["english"] < mean_e)] 

df_nc3 = df_exam[df_exam["nclass"] == 3]
df_nc3[["math", "english", "science"]]
df_nc3[0:1]
df_nc3[1:2]
df_nc3[1:5]


df_exam[0:10:2]
df_exam[7:16]

df_exam.sort_values("math", ascending=False)
df_exam.sort_values(["nclass", "math"], ascending=[True, False])


np.where(a > 3, "Up", "Down")
df_exam["updown"] = np.where(df_exam["math"] > 50, "Up", "Down") 
df_exam
