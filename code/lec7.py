fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]

# 빈 리스트 생성
empty_list1 = []
empty_list2 = list()

# 초기값을 가진 리스트 생성
numbers = [1, 2, 3, 4, 5]
range_list = list(range(5))
range_list

range_list[3] = "LS 빅데이터 스쿨"
range_list

#두번째 원소에 다음과 같은 정보를 넣어보자!
# ["1st", "2nd", "3rd"]
range_list[1] = ["1st", "2nd", "3rd"]
range_list

#"3rd"만 가져오고 싶다면
range_list[1][2]


# 리스트 내포(comprehension)
# 1. 대괄호로 쌓여져있다 => 리스트다.
# 2. 넣고 싶은 수식표현을 x를 사용해서 표현
# 3. for .. in .. 을 사용해서 원소정보 제공
list(range(10))
squares = [x**2 for x in range(10)]
squares

# 3, 5, 2, 15의 3제곱을 구하라 
my_squares = [x**3 for x in [3, 5, 2, 15]]
my_squares

# Numpy 어레이 와도 가능
import numpy as np
np.array([3, 5, 2, 15])
my_squares = [x**3 for x in np.array([3, 5, 2, 15])]
my_squares

# Pandas serise가 와도 가능?
import pandas as pd

exam = pd.read_csv("data/exam.csv")
exam["math"]
my_squares = [x**3 for x in exam["math"]]
my_squares

3 + 2
"안녕" + "하세요"
"안녕" * 3
# 리스트 연결
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined_list = list1 + list2

(list1 * 3) + (list2 * 5)


numbers = [5, 2, 3]
repeated_list = [x for x in numbers for _ in [4, 2, 1, 3]]
repeated_list = [x for x in numbers for _ in range(4)]
repeated_list = [x for x in numbers for _ in range(4)]
repeated_list = [x for x in numbers for y in range(4)]
repeated_list

# for 루프 문법
# for i in 범위:
# 작동방식
for x in [4, 1, 2, 3]:
    print(x)



mylist = []
mylist.append(2) # append = 리스트 끝에 원소 추가가
mylist.append(4)
mylist.append(6)
mylist
# 리스트를 하나 만들어서
# for 루프를 사용해서 2, 4, 6, 8, ..., 20의 수를 
# 채워넣어보세요!

[i for i in range(2, 21, 2)]
for x in range()

mylist = []
mylist = list(range(4))
mylist[0] = 2
mylist

# [ 0, 0, 0, 0] 만들기
mylist = [0] * 5
mylist

mylist = []
for i in range(1, 11):
    mylist.append(i*2)
mylist

mylist = [0] * 10
mylist[4] = 3 # 4번째 자리에 3 넣기

mylist = [0] * 10
for i in range(10):
    mylist[i] = (i + 1) * 2
mylist


# 퀴즈: mylist_b의 홀수번째 위치에 있는 숫자들만 mylist에
# 가져오기
mylist_b = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
mylist = [0] * 5
for i in range(5):
    mylist[i] = mylist_b[2*i]
mylist    


# 리스트 컴프리헨션으로 바꾸는 방법
# 바깥은 무조건 대괄호로 묶어줌: 리스트 반환하기 위해서.
# for 루프의 :는 생략한다.
# 실행부분을 먼저 써준다.
# 결과를 받는 부분 제외시킴
mylist = []
[i*2 for i in range(1, 11)]
[x for x in numbers]

for i in [0, 1]:
    for j in [4, 5, 6]:
        print(i, j)
    

for i in [0, 1]:
    for j in [4, 5, 6]:
        print(i)


numbers = [5, 2, 3]
for i in numbers:
    for j in range(4):
        print(i)


# 리스트 컴프리헨션 변환
[i for i in numbers for j in range(4)]

for x in numbers:
    for y in range(4):
        print(x)

# _ 의 의미
# 1. 앞에 나온 값을 가리킴
5 + 4
_ + 6 # _는 9를 의미

# 값 생략, 자리 차지(placeholder)
a, _, b = (1, 2, 4)
a; b
# _
# _ = None
# del _

for x in range(5) :
    print(x)


# 원소 체크
fruits = ["apple", "banana", "cherry"]
fruits
"banana" in fruits

# [x == "banana" for x in fruits]
mylist=[]
for x in fruits:
    mylist.append(x == "banana")
mylist

# 바나나의 위치를 뱉어내게 하려면?    
fruits = ["apple", "apple", "banana", "cherry"]

mylist=[]
for x in fruits:
    mylist.append(x == "banana")
my_index
    

# 바나나의 위치를 뱉어내게 하려면?
fruits = ["apple", "apple", "banana", "cherry"]

# int를 안썻는데 왜 그냥 2가 도출될까?
import numpy as np
fruits = np.array(fruits)
np.where(fruits == "banana")[0][0]

# 원소 거꾸로 써주는 reverse()
fruits = ["apple", "apple", "banana", "cherry"]
fruits.reverse()
fruits

# 원소 맨끝에 붙여주기
fruits.append("pineapple")
fruits

# 원소 삽입
fruits.insert(2, "test")
fruits


# 원소 제거
fruits.remove("test")
fruits

fruits.remove("apple")
fruits

import numpy as np

# 넘파이 배열 생성
fruits = np.array(["apple", "banana", "cherry", "apple", "pineapple"])

# 제거할 항목 리스트
item_to_remove = np.array(["banana", "apple"])

# 마스크(논리형 벡터) 생성
mask = ~np.isin(fruits, item_to_remove)

# 불리언 마스크를 사용하여 항목 제거
filtered_fruits = fruits[mask]
