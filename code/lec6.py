import numpy as np

# 벡터 슬라이싱 예제, a를 랜덤하게 채움
np.random.seed(2024)

a = np.random.randint(1, 21, 10)
print(a)

# 두 번째 값 추출
print(a[1])

a[-2] # 맨 끝에서 두번째째
a[::2]
a[0:6:2]

# 1에서부터 1000사이 3의 배수의 합은?
sum(np.arange(3,1001, 3))
x = np.arange(3,1001)
sum(x[2:1000:3])

print(a[[0, 2, 4]])

np.delete(a, [1, 3])

a > 3

b = a[a > 3]


print(b)

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
# a{조건을 만족하는 논리형벡터}
a[(a > 2000) & (a < 5000)]

!pip install pydataset
import pydataset

df = pydataset.data('mtcars')
np_df = np.array(df['mpg'])

model_names = np.array(df.index)
model_names[(np_df >= 15) & (np_df <= 25)]

# 15 이상 25 이하인 데이터 개수는?
sum((np_df >= 15) & (np_df <= 25))

# 15 이상 20 이하인 자동차 모델은?
model_names[(np_df >= 15) & (np_df <= 20)]

# 평균 mpg 보다 낮은(미만) 자동차 모델은은?
model_names[(np_df) < np.mean(np_df)]

# 15 작거나 25 이하인 데이터 개수는?
sum((np_df < 15) | (np_df >= 22))


np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
b = np.array(["A", "B", "C", "F", "W"])
# a{조건을 만족하는 논리형벡터}
a[(a > 2000) & (a < 5000)]
b[(a > 2000) & (a < 5000)]
a

a[ a > 3000] = 3000
a

np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)
a

# 처음으로 10000보다 큰 숫자가 나왔을 때,
# 그 숫자 위치와 그 숫자는 무엇인가요?
x=np.where(a > 22000)
type(x)
my_index =x[0][0]
a[my_index]

a[a > 5000][0]
a[np.where(a > 5000)]

a[np.where(a > 10000)][0]

# 처음으로 10000보다 큰 숫자 나왔을 때,
# 50번째로 나오는 숫자 위치와 그 숫자는 무엇인가요?
x = np.where(a > 10000)
a[x[0][49]]

# 500보다 작은 숫자들 중
# 가장 마지막으로 나오는 숫자 위치와 그 숫자는 무엇인가요?
x = np.where(a < 500)
x
a[x[0][-1]]


a = np.array([20, np.nan, 13, 24, 309])
np.isnan(a)
a + 3
np.mean(a)
np.nanmean(a)
np.nan +3
np.nan_to_num(a, nan = 0)

False
a = None
b = np.nan
b
a
b+1
a+1

a_filtered = a[~np.isnan(a)]
a_filtered


str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]]

mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
mix_vec

combined_vec = np.concatenate((str_vec, mix_vec))
combined_vec
# 'U2' = unicode라는 뜻

col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

row_stacked = np.vstack((np.arange(1, 5), np.arange(12, 16)))
row_stacked

uneven_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 18)))
uneven_stacked


vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)

np.resize(vec1,len(vec2))
vec1 = np.resize(vec1, len(vec2))
vec1

uneven_stacked = np.column_stack((vec1, vec2))
uneven_stacked

vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)

np.resize(vec1,len(vec2))

uneven_stacked = np.vstack((vec1, vec2))
uneven_stacked

# 홀수번째 원소
a = np.array([12, 21, 35, 48, 5])
a[0::2]

# 최댓값
a = np.array([1, 22, 93, 64, 54])
a.max()

a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a)

# 원소 번갈아가면서 합치기
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
a
b

x = np.empty(6)
x

# 짝수
# x[[1, 3, 5]] = b
x[1::2] = b
x

# 홀수
# x[[0, 2, 4]] = a
x[1::2] = b
x
