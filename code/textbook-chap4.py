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
type(df[["name"]])
type(df["name"])

# 평균
sum(df["english"])/4
