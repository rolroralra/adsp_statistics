import os  # 디렉토리 설정
import math  # 각종 수치연산(루트 등)
import numpy as np  # 선형대수, 행렬, 벡터
import pandas as pd  # CSV파일 읽기, DataFrame 객체, 평균, 중앙값, 분산, 표준편차, 사분위수, 상관관계
import matplotlib.pyplot as plt  # 박스플랏, 산점도
pd.set_option('display.max_columns', None) # show all columns when displaying dataset

#%%
##############################
# 1. 파이썬 기본함수
##############################
# 숫자형 연산
a = 3
b = 6

a + b
a - b
a * b
a / b
a ** b

(3 + (3 * (3 + 4)) - 5)
(10 - 4 * 2) ** 2 / (-2)

7 % 3
3 % 7
7 // 3
12 // 5

#%%
##############################
# 문자형
# indexing, slicing
a = 'My name is minsu kim'
a[0]
a[1]
a[-3]
a[0:1]
a[0:2]
a[:7]
a[0:]


#%%
##############################
# 리스트형
list1 = [ ]
list2 = [1, 2, 3]
list3 = ['a', 'b', 'c']
list4 = [1, 2, 'a', 'b']
list5 = [1, 2, 'a', 'b', [1, 2, 3]]
print( list1 )
print( list2 )
print( list3 )
print( list4 )
print( list5 )

   
# Convert list to DataFrame
list1 = [['Geeks',11],['For',22],['Geeks',33],['is',44],['portal',55],['for',66], ['Geeks',77]]   
df1 = pd.DataFrame(list1, columns =['Name', 'Age']) 
df1 
list1 = ['Geeks', 'For', 'Geeks', 'is', 'portal', 'for', 'Geeks'] 
list2 = [11, 22, 33, 44, 55, 66, 77] 

df2 = pd.DataFrame(list(zip(list1, list2)), columns =['Name', 'Age'], index=list(range(1,len(list1) + 1))) 
df2 


#%%
##############################
# 기본 함수
abs(-3.4)
max(2, 6, -9)
min(2, 6, -9)
pow(3, 3)
round(3.14159)
round(3.14159, 2)

# 수학 함수
math.factorial(5)
math.ceil(5.178)
math.floor(5.678)
math.exp(-1)
math.log(1)
math.log2(8)
math.log10(100)
math.sqrt(16)
np.log([1,2])
np.sqrt([1,2])


#%%
##############################
# 2. 데이터 불러오기
##############################
# 디렉토리 설정
os.getcwd()
os.chdir('K:\\My files\\00. 교육자료(개인)\\DS\\datasets_김현중_200629')
os.getcwd()

##############################
# read excel file
usedcar = pd.read_excel('usedcar2.xlsx') 
usedcar
# read text file with separator
hmeq = pd.read_csv('hmeq.txt', sep="\t")
hmeq
# read csv file
usedcar2 = pd.read_csv('usedcar2.csv') 
usedcar2

iris=pd.read_csv('iris.txt')
iris


#%%
##############################
# 3. 데이터를 다루기 위한 Python 함수
##############################
# 데이터 속성 파악
usedcar2.shape
usedcar2.dtypes
# 특정위치 데이터 추출
# Use `iloc[]` to select a row
usedcar2.iloc[3]
# Use `[]` to select a column
usedcar2['Color']
# 특정 row, column을 선택하기
usedcar2.iloc[50]['Color']
usedcar2.iloc[50,2]
usedcar2.iloc[50][2]

# delete column
usedcar_reduce = usedcar2.drop('Color', axis=1) # axis='columns' also working
# usedcar_reduce = usedcar2.drop('Color', axis='columns')
usedcar_reduce
# delete row
usedcar_reduce = usedcar2.drop(0, axis=0) # axix='rows' also working
# usedcar_reduce = usedcar2.drop(0, axis='rows') 
usedcar_reduce


#%%
##############################
# 4. 정규화 및 표준화
##############################
from sklearn.preprocessing import MinMaxScaler    # sklearn: scikit-learn
from sklearn.preprocessing import StandardScaler
normalize = MinMaxScaler()
standardize = StandardScaler()

hmeq = pd.read_csv('hmeq.txt', sep="\t")
hmeq

# 정규화
normalize.fit(hmeq[['MORTDUE']])  # use [['MORTDUE']] to make dataframe
hmeq['norm'] = normalize.transform(hmeq[['MORTDUE']]) 
hmeq.describe()
# 표준화
standardize.fit(hmeq[['MORTDUE']])
hmeq['stan'] = standardize.transform(hmeq[['MORTDUE']]) 
hmeq.describe()
# 로그/루트 변환
hmeq['log'] = np.log(hmeq['MORTDUE']) 
hmeq['sqrt'] = np.sqrt(hmeq['MORTDUE']) 
hmeq.describe()
plt.hist(hmeq['MORTDUE'], bins=20)
plt.hist(hmeq['log'], bins=20)
plt.hist(hmeq['sqrt'], bins=20)


#%%
##############################
# 5. 변수이름 바꾸기 및 새로운 변수 추가하기
##############################
# 변수이름 바꾸기
sales = pd.read_csv('sales.csv')
sales.columns
sales.columns = ['city','district','gender','store.name','store.code','ymd','sales.total']
sales
sales.dtypes

# 변수 생성 및 변경
# ymd를 string 형으로
sales['ymd'] = sales['ymd'].apply(str)
# ymd를 이용하여 ym 생성
sales['ym'] = sales['ymd'].str[0:6]
# store.code 이용하여 biz.cd 생성
sales['biz.code'] = sales['store.code'].str[0:4]
# 불필요 변수 제거
sales = sales.drop('city', axis=1)
sales = sales.drop('store.code', axis=1)


#%%
##############################
# 6. 데이터 정렬 및 조건에 맞는 데이터 추출하기
##############################

# 판매건수 기준 내람차순 정렬
sales_sort = sales.sort_values('sales.total', ascending=False)
sales_sort.head(10)

# 판매건수와 구별 내림차순 정렬
sales_sort = sales.sort_values(['sales.total','district'], ascending=[False,False])
sales_sort

# 특정조건 맞는 데이터 추출
sales03 = sales[sales['ym']=='201403']
sales03


#%%
##############################
# 7. 데이터 요약하기
##############################
# data summary
sales['store.name'].value_counts()
sales
sales.dtypes
sales.describe()
pd.crosstab(sales['district'],sales['store.name'])

sales.groupby('store.name').count()

# 변수별 결측치 개수
sales.isnull().sum()
sales['gender'].value_counts(dropna=False)

# group 별 계산
sales.groupby('district').get_group('강남구')
sales.groupby('district').mean()
sales.groupby('district').sum()
sales.dtypes
sales[['district','sales.total']].groupby('district').sum()

sales_ymstore = sales.groupby(['district','ym','store.name']).sum()
sales_ymstore.head(30)

# 판매점들의 월판매량 내림차순으로 정렬
sales_ymstore_sort = sales_ymstore.sort_values(['sales.total'], ascending=False)
sales_ymstore_sort.head(20)

# 송파구 연월별 판매점들의 판매량 내림차순으로 정렬
sales_songpa = sales[(sales['store.name']!='@') & (sales['district']=='송파구')]
sales_songpa1 = sales_songpa.groupby(['ym','store.name']).sum()
sales_songpa1.head(30)
sales_songpa1_sort = sales_songpa1.sort_values(['sales.total'], ascending=False)
sales_songpa1_sort.head(10)


#%%
##############################
# 8. 다른 데이터 합치기
##############################
list1 = ['카페', '한식', '유아용품', '양식', '전자제품'] 
list2 = [1,2,3,4,5] 
df1 = pd.DataFrame(list(zip(list1, list2)), columns =['title', 'rank1']) 
df1 
list1 = ['한식', '양식', '전자제품', '일식', '꽃'] 
list2 = [1,2,3,4,5] 
df2 = pd.DataFrame(list(zip(list1, list2)), columns =['title', 'rank2']) 
df2 

pd.merge(df1, df2, how='left', on='title')
pd.merge(df1, df2, how='right', on='title')
pd.merge(df1, df2, how='outer', on='title')
pd.merge(df1, df2, how='inner', on='title')

# product code data
products = pd.read_csv('productcode.csv')
products.columns
products.columns = ['biz.code','product.name']
products
sales
merge_data1 = pd.merge(sales, products, how='left', on='biz.code')
merge_data1

# climate data
climates = pd.read_csv('seoul_climate.csv')
climates.columns
climates.columns = ['city','district','ym','latitude','longitude','rainfall']
climates.dtypes
climates['ym'] = climates['ym'].apply(str)
climates.dtypes
climates = climates.drop('city', axis=1)
climates

merge_data2 = pd.merge(merge_data1, climates, how='left') 
merge_data2

#한식 업종에서 점포별로 일 평균 판매건수
# 한식업종 데이터 추출
hansik = merge_data2[merge_data2['product.name']=='한식']
hansik
# 일평균 판매건수
hansik_store = hansik.groupby(['store.name']).mean()
hansik_store

# 강수량 데이터에서 250mm이상은 ‘상‘, 50mm~250mm는 ‘중‘, 50mm 이하는 ‘하’로 변환
merge_data2.columns
merge_data2.describe()
bins = [0,50,250,300]
labels = ["하", "중", "상"]
# right=False (이상~미만)
category = pd.cut(merge_data2['rainfall'], bins, labels=labels, right=False)
type(category)
type(pd.DataFrame(category))
category
pd.DataFrame(category)

category.value_counts(dropna=False)
merge_data2['rain_cat'] = pd.DataFrame(category)
merge_data2.max()
sales_rain = merge_data2[['ym','product.name','rain_cat','sales.total']]
sales_rain.dtypes

# 강수량 ‘상’,’중’,’하’에 따른 업종별 일평균 판매건수
sales_rain_mean = sales_rain.groupby(['product.name','rain_cat']).mean()
sales_rain_mean


#%%
##############################
# 9. 결측치 처리하기
##############################
# 결측치 제거 혹은 채워넣기
hmeq = pd.read_csv('hmeq.txt', sep="\t")
hmeq
df = hmeq.iloc[:,6:]  # 모든 행, 6번째 컬럼이후만
df
# 변수별 결측치 비율
df.isnull().sum() / len(df)

# 결측 관찰치 제거
df_cleaned = df.dropna()
df_cleaned.head()
df_cleaned.shape

# 결측치 0 혹은 다른 숫자로 채우기
df_impute0 = df.fillna(0)
df_impute0

# 결측치 mean/median로 채우기
df.describe()
df_impute_mean = df.fillna(df.mean())
df_impute_mean.head()
df_impute_median = df.fillna(df.median())
df_impute_median.head()


#%%
##############################
# 10. 파일로 저장하기
##############################
# Read/Write dataset
hmeq = pd.read_csv('hmeq.txt', sep='\t')
hmeq.to_csv('hmeq.csv', index=False)
hmeq.to_excel('hmeq.xlsx', index=False)

build_data = pd.read_csv("building.csv") # not working
build_data = pd.read_csv("building.csv", encoding="EUC-KR")
build_data.to_csv('building_euc.csv', index=False, encoding="EUC-KR") 
build_data.to_csv('building_utf.csv', index=False)
build_data.to_excel('building.xlsx', index=False)

merge_data2.to_csv('sales_euc.csv', index=False, encoding="EUC-KR") 
merge_data2.to_excel('sales.xlsx', index=False)
