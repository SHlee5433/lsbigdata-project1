# 데이터 타입
x = 15.34
print(x, "는", type(x), "형식입니다.", sep='')


# 문자형 데이터 예제
a = "Hello, World!"
b = "python programming"

print(a, type(a))
print(b, type(b))



# 문자열 결합
greeting = "안녕" + " " + "파이썬!"
print("결합 된 문자열:", greeting)

# 문자열 반복
laugh = "하" * 3
print("반복 문자열:", laugh)

# 리스트
fruit = ['apple', 'banana', 'cherry']
type(fruit)

numbers = [1, 2, 3, 4, 5]
type(numbers)

mixed_list = [1, "Hello", [1, 2, 3]]
type(mixed_list)

a_ls = [10, 20, 30, 40, 50]
a_ls[1:4]
a_ls[:3]
a_ls[1] = 25
a_ls

a_tp = (10, 20, 30, 40, 50) # a = 10, 20, 30 과 동일
a_tp[3:] # 해당 인덱스 이상
a_tp[:3] # 해당 인덱스 미만만
a_tp[1:3] # 해당 인덱스 이상 & 미만만

a = (10, 20, 30) # a = 10, 20, 30 과 동일
a[0]
a[1]

b_int = (42)
b_int
type(b_int)
b_tp = (42,)
b_tp


# 사용자 정의함수
def min_max(numbers):
  return min(numbers), max(numbers)

a = [1, 2, 3, 4, 5]
result = min_max(a)
result
print("minimun and maximum:", result)

# 딕셔너리

person = {
  'name': 'John',
  'age': 30,
  'city': 'New York'
}

issac = {
  'name': '이삭',
  '나이': (39,30),
  '사는곳': ['미국', '한국']
}
print("Person:", person)
print("Issac:", issac)

issac_age=issac.get('사는곳')
issac_age[0]

# 집합
fruits = {'apple', 'banana', 'cherry', 'apple'}
print("Fruits set:", fruits) # 중복 'apple'은 제거됨
type(fruit)

# 빈 집합 생성
empty_set = set()
print("Empty set:", empty_set)

empty_set.add('apple')
empty_set.add('banana')
empty_set.remove('banana')
empty_set.discard('banana')

empty_set


# 집합 간 연산
other_fruits = {'berry', 'cherry'}
union_fruits = fruits.union(other_fruits) # 합집합
intersection_fruits = fruits.intersection(other_fruits) #교집합
print("Union of fruits:", union_fruits)
print("Intersection of fruits:", intersection_fruits)


# 논리형 데이터 예제
p = True
q = False
print(p, type(p))
print(q, type(q))
print(p + p)

age = 10
is_active = True
is_greater = age > 5 # True 반환
is_equal = (10 == 5) # False 반환
print("Is active:", is_active)
print("Is 10 greater than 5?:", is_greater)
print("Is 10 equal to 5?:", is_equal)

a=3
if (a == 2) :
  print("a는 2와 같습니다.")
else:
  print("a는 2와 같지 않습니다.")


# 숫자형을 문자열형으로 변환
num = 123
str_num = str(num)
print("문자열:", str_num, type(str_num))

# 문자열형을 숫자형(실수)으로 변환
num_again = float(str_num)
print("숫자형:", num_again, type(num_again))


# 리스트와 튜플 변환
lst = [1, 2, 3]
print("리스트:", lst)
tup = tuple(lst)
print("튜플:", tup)


set_example = {'a', 'b', 'c'}
# 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example}
print("Dictionary from set:", dict_from_set)
