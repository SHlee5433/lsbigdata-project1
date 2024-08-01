!pip install pyreadstat
import pandas as pd
import numpy as np
import seaborn as sns

raw_welfare = pd.read_spss("./data/Koweps_hpwc14_2019_beta2.sav")
raw_welfare

welfare = raw_welfare.copy()
welfare.shape
welfare.describe()

welfare = welfare.rename(
    columns = {
        "h14_g3" : "sex",
        "h14_g4" : "birth",
        "h14_g10" : "marriage_type",
        "h14_g11" : "religion",
        "p1402_8aq1" : "income",
        "h14_eco9" : "code_job",
        "h14_reg7" : "code_region"
    }
)

welfare = welfare[["sex", "birth", "marriage_type",
                "religion", "income", "code_job", "code_region"]]
welfare

welfare["sex"].dtypes
welfare["sex"].value_counts()

welfare["sex"] = np.where(welfare["sex"] == 1, "male", "female")


welfare["income"].describe()
welfare["income"].isna().sum()

# 성별 수입의 평균
sex_income = welfare.dropna(subset = "income") \
                    .groupby("sex", as_index = False) \
                    .agg(mean_income = ("income", "mean"))
sex_income

import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(data = sex_income, x = "sex", y = "mean_income",
            hue = "sex")
plt.show()
plt.clf()

# 숙제 : 위 그래프에서 각 성별 95% 신뢰구간 계산후 그리기
# 위 아래 검정색 막대기로 표시

welfare["birth"].describe()
sns.histplot(data = welfare, x = "birth")
plt.show()
plt.clf()

welfare["birth"].isna().sum()

welfare = welfare.assign(age = 2019 - welfare["birth"] + 1)
welfare["age"]
sns.histplot(data = welfare, x = "age")
plt.show()
plt.clf()

age_income = welfare.dropna(subset = "income") \
                    .groupby("age") \
                    .agg(mean_income = ("income", "mean"))
age_income.head()

sns.lineplot(data = age_income, x = "age", y = "mean_income")
plt.show()
plt.clf()

# 나이별 income 칼럼 na 개수 세기!
welfare["income"].isna().sum()

welfare["income"].isna()

my_df = welfare.assign(income_na = welfare["income"].isna()) \
                .groupby("age", as_index = False) \
                .agg(n = ("income_na", "count"))
                
                
sns.barplot(data = my_df, x = "age", y = "n")
plt.show()
plt.clf()

welfare["age"].head()
welfare = welfare.assign(ageg = np.where(welfare["age"] < 30, "young",
                                np.where(welfare["age"] <= 59, "middle",
                                                              "old")))
                                                              
welfare["ageg"].value_counts()                                                        

sns.countplot(data = welfare, x = "ageg", hue = "ageg")
plt.show()
plt.clf()

ageg_income = welfare.dropna(subset = "income") \
                     .groupby("ageg", as_index = False) \
                     .agg(mean_income = ("income", "mean"))

ageg_income

sns.barplot(data = ageg_income, x = "ageg", y = "mean_income", hue = "ageg")
plt.show()
plt.clf()

sns.barplot(data = ageg_income, x = "ageg", y = "mean_income",
            order = ["young", "middle", "old"], hue = "ageg")
plt.show()
plt.clf()

# 나이대별 수입 분석
# cut

# 내가 짠 코드
vec_x = np.random.randint(0, 100, 50)

age_min, age_max = (welfare['age'].min(), welfare['age'].max())
age_max // 10

bin_cut = [0] + [10 * i + 9 for i in np.arange(age_max//10 + 1)]
pd.cut(vec_x, bins=bin_cut)

# 유용하게 쓰일 코드!!
bin_cut = np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
welfare = welfare.assign(age_group = pd.cut(welfare["age"],
                bins = bin_cut,
                labels = (np.arange(12) * 10).astype(str) + "대"))

age_income = welfare.dropna(subset = "income") \
                     .groupby("age_group", as_index = False) \
                     .agg(mean_income = ("income", "mean"))
    
age_income

sns.barplot(data = age_income, x = "age_group", y = "mean_income")
plt.show()
plt.clf()

# 판다스 데이터 프레임을 다룰 때, 변수의 타입이
# 카테고리로 설정되어 있는 경우, groupby + agg 콤보
# 안먹힘. 그래서 object 타입으로 바꿔준 후 수행
welfare["age_group"] = welfare["age_group"].astype("object")

def my_f(vec):
    return vec.sum()

sex_age_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group", "sex"], as_index = False) \
    .agg(top4per_income = ("income", lambda x: my_f(x)))

sex_age_income


sex_age_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group", "sex"], as_index = False) \
    .agg(top4per_income = ("income", lambda x: np.quantile(x, q = 0.96)))

sex_age_income


sns.barplot(data = sex_age_income,
            x = "age_group", y = "mean_income",
            hue = "sex")
plt.show()
plt.clf()

# 연령대별, 성별 상위 4% 수입 찾아보세요!

import pandas as pd

# 예제 데이터 프레임 생성
data = {
    'income': [50000, 60000, 70000, None, 80000, 90000, None, 100000, 110000],
    'age_group': ['20대', '30대', '40대', '20대', '30대', '40대', None, '20대', '30대'],
    'sex': ['M', 'F', 'M', 'F', None, 'M', 'F', 'M', 'F']
}
welfare = pd.DataFrame(data)

