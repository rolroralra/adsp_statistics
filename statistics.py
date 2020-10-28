"""
Statistics using Python by Prof. Uk Jung
1. 통계학 (Statistics)
2. 기술통계학 (Descriptive Statistics)
3. 확률분포 (Probability Distribution)
4. 확률표본과 표집분포 (Random Sample & Sampling Distribution)
5. 추론통계학: 추정 (Interential Statistics: Estimation)
6. 추론통계학: 가설검정 (Inferential Statistics: Hypothesis Test)
7. 범주형 자료분석 (Categorical Data Analysis)
8. 분산분석 (Analysis of Variance; ANOVA)
9. 상관분석과 회귀분석 (Correlation Analysis & Regression Analysis)
"""
#%%
"""Statistics using Python by Prof. Uk Jung
2. 기술통계학 (Descriptive Statistics)
"""
#%% 그래프 기본
import matplotlib.pyplot as plt

#%% 그래프 기본: 1) y 값만 있는 경우
data1 = [10, 14, 19, 20, 25]

plt.plot(data1)
plt.show()
#%% 그래프 기본: 2) x값, y값이 모두 있는 경우
import numpy as np

x = np.arange(-4.5, 5, 0.5) # 배열 x 생성. 범위: [-4.5, 5), 0.5씩 증가
y = 2*x**2 # 수식을 이용해 배열 x에 대응하는 배열 y 생성
[x,y]

plt.plot(x,y)
plt.show()

#%% 그래프 기본: 3) 하나의 창에 다수의 그래프 그리기
import numpy as np

x = np.arange(-4.5, 5, 0.5)
y1 = 2*x**2
y2 = 5*x + 30
y3 = 4*x**2 + 10

import matplotlib.pyplot as plt

plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.show()

#plt.plot(x, y1, x, y2, x, y3)
#plt.show()
#%% 그래프 기본: 4) 새로운 그래프 창에 개별적인 그래프 그리기
plt.plot(x, y1) #처음 그리기 함수를 수행하면 그래프 창이 자동으로 생성됨

plt.figure() # 새로운 그래프 창을 생성함
plt.plot(x, y2) # 새롭게 생성된 그래프 창에 그래프를 그림

plt.show()

#%% 그래프 기본: 5) 개별적인 그래프 창에 그래프 따로 그리기
import numpy as np

# 데이터 생성
x = np.arange(-5, 5, 0.1)
y1 = x**2 -2
y2 = 20*np.cos(x)**2 # NumPy에서 cos()는 np.cos()으로 입력

plt.figure(1) # 1번 그래프 창을 생성함
plt.plot(x, y1) # 지정된 그래프 창에 그래프를 그림

plt.figure(2) # 2번 그래프 창을 생성함
plt.plot(x, y2) # 지정된 그래프 창에 그래프를 그림

plt.figure(1) # 이미 생성된 1번 그래프 창을 지정함
plt.plot(x, y2) # 지정된 그래프 창에 그래프를 그림

plt.figure(2) # 이미 생성된 2번 그래프 창을 지정함
plt.clf() # 2번 그래프 창에 그려진 모든 그래프를 지움
plt.plot(x, y1) # 지정된 그래프 창에 그래프를 그림

plt.show()

#%% 그래프 기본: 6) 하나의 그래프 창을 하위 그래프 영역으로 나누기
import numpy as np

# 데이터 생성
x = np.arange(0, 10, 0.1)
y1 = 0.3*(x-5)**2 + 1
y2 = -1.5*x + 3
y3 = np.sin(x)**2 # NumPy에서 sin()은 np.sin()으로 입력
y4 = 10*np.exp(-x) + 1 # NumPy에서 exp()는 np.exp()로 입력

# 2 × 2 행렬로 이뤄진 하위 그래프에서 p에 따라 위치를 지정
plt.subplot(2,2,1) # p는 1
plt.plot(x,y1)
plt.subplot(2,2,2) # p는 2
plt.plot(x,y2)
plt.subplot(2,2,3) # p는 3
plt.plot(x,y3)
plt.subplot(2,2,4) # p는 4
plt.plot(x,y4)

plt.show()

#%% 그래프 기본: 7) 그래프의 출력 범위 지정하기
import numpy as np

x = np.linspace(-4, 4,100) # [-4, 4] 범위에서 100개의 값 생성
y1 = x**3  
y2 = 10*x**2 - 2

plt.plot(x, y1, x, y2)
plt.xlim(-1, 1)
plt.ylim(-3, 3)
plt.show()

#%% 그래프 기본: 8) 출력 형식 지정 (format string(fmt))
import numpy as np
x = np.arange(0, 5, 1)
y1 = x
y2 = x + 1
y3 = x + 2
y4 = x + 3

#plt.plot(x, y1, x, y2, x, y3, x, y4)
#plt.plot(x, y1, 'm', x, y2,'y', x, y3, 'k', x, y4, 'c')
#plt.plot(x, y1, '-', x, y2, '--', x, y3, ':',  x, y4, '-.')
#plt.plot(x, y1, 'o', x, y2, '^',x, y3, 's', x, y4, 'd')

plt.plot(x, y1, '>--r', x, y2, 's-g', x, y3, 'd:b', x, y4, 'x-.c')
plt.show()

#%% 그래프 기본: 9) Label, Title, Grid, Legend, Text 표시
import numpy as np

x = np.arange(-4.5, 5, 0.5)
y1 = 2*x**3
y2 = x + 1

plt.plot(x, y1, '>--r', x, y2, 's-g')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Graph title')
plt.grid(True) # 'plt.grid()'도 가능
plt.legend(['korea', 'y2'], loc = 'best') # loc = 'lower right', ...
plt.text(-3, 100, "young 1")
plt.text(0, 25, "old 2")
plt.show()

#%% 그래프 기본: 10) 한글 폰트 지정
import matplotlib

matplotlib.rcParams['font.family'] = 'Malgun Gothic'   # '맑은 고딕'으로 설정 
matplotlib.rcParams['axes.unicode_minus'] = False

plt.plot(x, y1, '>--r', x, y2, 's-g')
plt.text(-3, 100, "문자열 출력 1")
plt.text(0, 25, "문자열 출력 2")
plt.show()

#%% Python을 이용한 시각화 (barplot)

member_IDs = ['m_01', 'm_02', 'm_03', 'm_04'] # 회원 ID
before_ex = [27, 35, 40, 33] # 운동 시작 전
after_ex = [30, 38, 42, 37] # 운동 한 달 후

import matplotlib.pyplot as plt
import numpy as np

n_data = len(member_IDs)     # 회원이 네 명이므로 전체 데이터 수는 4
index = np.arange(n_data)   # NumPy를 이용해 배열 생성 (0, 1, 2, 3)
colors=['r', 'g', 'b', 'm']
#plt.bar(index, before_ex)
plt.bar(index, before_ex, 
        color = colors, 
        tick_label = member_IDs, 
        width = 0.2)   # bar(x,y)에서 x=index, y=before_ex 로 지정

#plt.barh(index, before_ex)
plt.show()

#%% hangeul font 
from matplotlib import rc
font_name = 'Malgun Gothic'
rc('font', family = font_name)

#%% bar plot with two groups 
barWidth = 0.4
plt.bar(index, before_ex, 
        color='c', align='edge', 
        width = barWidth,label='before')
plt.bar(index + barWidth, after_ex , color='m', align='edge', 
		width = barWidth, label='after')

plt.xticks(index + barWidth, member_IDs)
plt.legend()
plt.xlabel('회원 ID')
plt.ylabel('윗몸일으키기 횟수')
plt.title('운동 시작 전과 후의 근지구력(복근) 변화 비교')
plt.show()

#%%#%% Python을 이용한 시각화 (pie)

fruit = ['사과', '바나나', '딸기', '오렌지', '포도']
result = [7, 6, 3, 2, 2]

import matplotlib.pyplot as plt

#plt.figure(figsize=(5,5))
#plt.pie(result)

explode_value=(0.5, 0,0,0,0)
plt.pie(result, 
        labels= fruit, autopct='%.1f%%', 
        startangle=90, counterclock = False, 
        explode=explode_value, shadow=True)
plt.show()

#%% Python을 이용한 시각화 (histogram)

import matplotlib.pyplot as plt

math = [76, 82, 84, 83, 90, 86, 85, 92, 72, 71, 
        100, 87, 81, 76, 94, 78, 81, 60, 79, 69, 
        74, 87, 82, 68, 79]
#plt.hist(math)
plt.hist(math, bins= 4)

plt.xlabel('시험 점수')
plt.ylabel('도수(frequency)')
plt.title('수학 시험의 히스토그램')
#plt.grid()
plt.show()
#%% Python을 이용한 시각화(scatter plot)

import matplotlib.pyplot as plt

height = [179, 173, 165, 177, 160, 180, 185, 155, 172, 165]   # 키 데이터
weight = [80, 70, 62, 67, 55, 74, 90, 43, 64,75]   # 몸무게 데이터

plt.scatter(height, weight)
plt.xlabel('Height(m)')
plt.ylabel('Weight(Kg)')
plt.title('Height & Weight')
#plt.grid(True)

#%% Python을 이용한 시각화 (boxplot)
import seaborn as sns
ax = sns.boxplot(x=before_ex)
#%% multiple boxplots
list=[before_ex, after_ex]

import numpy as np
table=np.transpose(list)

import pandas as pd
df_table=pd.DataFrame(table, columns=['before_ex', 'after_ex'])

import seaborn as sns
ax = sns.boxplot(data=df_table, 
			   orient="h",
			   palette="Set2")
#%% 중심위치의 측도 
height_mean = np.mean(height)
height_median = np.median(height)
print("mean:", height_mean)
print("median:", height_median)
#%% Mode
import statistics
print("mode:",statistics.mode(height))
#%% 산포의 측도 
# var
height_var = np.var(height)
# sample var
height_var_s = np.var(height, ddof = 1)
print("var:", round(height_var, 2))
print("sample var:", round(height_var_s, 2))

# std
height_std = np.std(height)
# sample std
height_std_s = np.std(height, ddof = 1)
print("std:", round(height_std, 2))
print("sample std:", round(height_std_s, 2))

# range
height_range = np.max(height) - np.min(height)
print("range:", height_range)

# IQR
q1, q3 = np.percentile(height, [25, 75])
height_IQR = q3 - q1
print("IQR:", round(height_IQR, 2))
#%% 분포의 형태(shape)에 관한 측도
from scipy.stats import skew
from scipy.stats import kurtosis
print("skewness:",round(skew(height), 2))
print("kurtosis",round(kurtosis(height), 2))
#%% Python에서 자료의 요약을 보여주는 describe() 함수
import pandas as pd
height_df = pd.DataFrame(height)
summary = height_df.describe()
print(summary)
#%%
""" Statistics using Python by Prof. Uk Jung
3. 확률분포 (Probability Distribution)
"""
#%% 상관계수
corr=np.corrcoef(height, weight)[0][1]
print("correlation coefficient:", round(corr, 3))
#%% 이산형 확률분포함수 (이항분포, 포아송 분포)
# calculate cdf of binomial
from scipy import stats
bin_cdf=stats.binom.cdf(10,200, 0.04)
print("P(x<=10) when x~bin(200, 0.04):", round(bin_cdf,3))

# calculate cdf of poisson
poi_cdf=stats.poisson.cdf(10,8)
print("P(x<=10) when x~Poisson(8):", round(poi_cdf, 3))
#%% 연속형 확률분포함수 (정규분포) 
#pdf, cdf of normal dist

from scipy.stats import norm   
loc=3; scale=2; x=2.5
print("normal pdf=",round(norm.pdf(x, loc, scale), 3))
print("normal cdf=",round(norm.cdf(x, loc, scale), 3))
p=0.3
print("normal quantile=", round(norm.ppf(p,loc,scale), 3))
print("normal randome variables=", norm.rvs(loc, scale, size=5))
#%% Python 실습 – t, 카이제곱, F 분포
# pdf, cdf, quantile of chisquare dist

from scipy.stats import chi2
# degree of freedom
dof=3
aaa=chi2.cdf(2, dof)
print("chi-square cdf=",round(aaa,3))
#print("chi-square quantile=",chi2.ppf(0.95, dof))

# pdf, cdf, quantile of t dist
from scipy.stats import t
dof=10
print("t pdf=",t.pdf(-1, dof))
print("t quantile=",t.ppf(0.5, dof))
print("t random variables=", t.rvs(dof, size=5))

# pdf, cdf, quantile of F dist
from scipy.stats import f
dof1=5
dof2=7
aaa=f.cdf(3, dof1, dof2)
print("F cdf=",round(aaa,3))
#%%
"""Statistics using Python by Prof. Uk Jung
4. 확률표본과 표집분포 (Random Sample & Sampling Distribution)
"""
#%%표본평균의 히스토그램
#mean of random sample
import numpy as np
sample_array = []
random_seed=1234
np.random.seed(random_seed)

min_value=0
max_value=10
sample_size=100
# number_of_samples=5
# number_of_samples=30
number_of_samples=100

for i in range(number_of_samples):
    sample = np.random.randint(min_value, max_value, size = sample_size)
    sample_array.append(np.mean(sample))
sample_array = np.array(sample_array)
print("samples:",sample_array)

# histogram
import matplotlib.pyplot as plt
plt.hist(sample_array, bins = number_of_samples, range=[min_value, max_value])
plt.xlabel('sample_mean'); plt.ylabel('Frequency')
plt.title('Historam of sample_mean')
plt.show()
#%%
"""Statistics using Python by Prof. Uk Jung
5. 추론통계학: 추정 (Interential Statistics: Estimation)
"""
#%% 신뢰구간을 활용한 모평균 추정
# data input for Battery Life
import numpy as np
battery_life = np.array([260, 265, 250, 270, 272, 
                         258, 262, 268, 270, 252])

    # descriptive summary
xbar_b = np.mean(battery_life); print("xbar:", xbar_b)
svar_b = np.var(battery_life, ddof=1); print("sample var:", round(svar_b,3))
ssd_b = np.std(battery_life, ddof=1); print("sample sd:", round(ssd_b,3))
n = battery_life.size; print("n:", n)
#%% confidence interval
# standard error
import math  
from scipy import stats
se_b = ssd_b/math.sqrt(n)
print("standard error:", round(se_b,3))
#from scipy import stats
#se_b = stats.sem(battery_life)

# t_alpha with dof=n-1 and alpha=0.05
t_alpha = stats.t.ppf(1-0.05/2, n-1)
print("t alpha:", round(t_alpha,3))

# error margin
itval = t_alpha*se_b
print("error margin:", round(itval,3))

# confidence interval
CI = [round(xbar_b-itval,3), round(xbar_b+itval,3)]
print("95% confidence interval:", CI)
#%%
"""Statistics using Python by Prof. Uk Jung
6. 추론통계학: 가설검정 (Inferential Statistics: Hypothesis Test)
"""
#%% 모평균에 관한 검정: battery

# test statistics & P-value for two-side(H1: mu is not equal to 257)

tval = (xbar_b-257)/se_b; print("t statistic:", round(tval,3))
pval = 2*(1-stats.t.cdf(np.abs(tval), n-1)); print("P-value:", round(pval,3))
#stats.ttest_1samp(battery, popmean=257)
#%% test statistics & P-value for one-side(H1: mu<257)
tval = (xbar_b-257)/se_b; print("t statistic:", round(tval,3))
pval = stats.t.cdf(tval, n-1); print("P-value:", round(pval,3))
#%% 모평균에 관한 검정: cellphone 
# Read CSV file
import pandas as pd
# 데이터셋 부르기
# 방법1: 디렉토리 주소를 포함하여 파일명 적어주기
# 방법2: 현재의 working directory에 파일을 옮겨놓기
# 방법3: 파일이 저장된 디렉토리를 working directory로 정하기 ****
cell=pd.read_csv('./data/cellphone.csv')
X=cell['AMOUNT']

xbar_b = np.mean(X); print("x_bar:", xbar_b)
#var_b = np.var(X, ddof=1); print("sameple var:", round(var_b,3))
sd_b = np.std(X, ddof=1); print("sample sd:", round(sd_b,3))
n = X.size; print("n:", n)
#%% test statistics & P-value for one-side(H1: mu>55000)

from scipy import stats
# standard error
se_b = stats.sem(X); print("standard error:", round(se_b,3))

# test statistics & P-value 
tval = (xbar_b-55000)/se_b; print("t statistic:", round(tval,3))
pval = 1-stats.t.cdf(tval, n-1); print("P-value:", round(pval,3))
#%% 모분산에 관한 검정: popcorn

import numpy as np
popcorn = np.array([198, 201, 199, 189, 200,
                    199, 198, 189, 205, 195])
sd_b = np.std(popcorn, ddof=1); print("sd:", round(sd_b,3))
n = popcorn.size; print("n:", n)
 
from scipy.stats import chi2
df=n-1; print("chi-square 0.95 quantile=",round(chi2.ppf(0.95, df),3))
test_stat=(n-1)* sd_b**2/25; print("test stat=", round(test_stat,3))

# p-value for H1: sigma_square > 25
print("P-value=",round(1-chi2.cdf(test_stat, df),3))
#%% 두 모집단의 모평균 차이에 관한 검정: whours
import pandas as pd
whours=pd.read_csv('./data/whours.csv')

whours_M = whours[whours['gender']=='M'].hours 
whours_F = whours[whours['gender']=='F'].hours

#import scipy.stats 
results = stats.ttest_ind(whours_M, whours_F, equal_var=True)
print("test statistic: ", round(results[0],3))
print("p-value: ", round(results[1],3))
print("p-value: ", round(1- t.cdf(results[0], len(whours_F) - 1), 3))
#%% 대응표본에 대한 모평균 차이에 관한 검정: salespairs
import pandas as pd
spairs=pd.read_csv('./data/salespairs.csv')

sales_before = spairs[spairs['status']==1].sales.values
sales_after = spairs[spairs['status']==2].sales.values
sales_diff = sales_after - sales_before

from scipy.stats import t
import numpy as np 
import scipy.stats as stats
alpha=0.05
n = len(sales_diff); d_of_f= n - 1
print("degrees of freedom: {0}".format(d_of_f))

t_critical =round(t.ppf(q=1-alpha, df=d_of_f), 2)
print("t-critical value of {0}".format(t_critical))
#%%
mean_p_est = np.mean(sales_after) - np.mean(sales_before)
std_sales_diff = np.std(sales_diff, ddof=1)

t_stat = mean_p_est/(std_sales_diff/np.sqrt(n))
print("t value:", round(t_stat,3))

pval = 1-stats.t.cdf(t_stat, n-1); print("P-value:", round(pval,3))

# For two-tail test, use stats.ttest_rel: 
# stats.ttest_rel(sales_after, sales_before)
#%%
"""Statistics using Python by Prof. Uk Jung
7. 범주형 자료분석 (Categorical Data Analysis)
"""
#%% 카이제곱 적합성 검정: 라면시장

# data input
import numpy as np

Obs = np.array([285, 66, 55, 44])
Pr = np.array([0.68, 0.13, 0.11, 0.08])
n = Obs.sum(); E = n*Pr; df = len(Obs)-1

# chi-squared goodness of fit test(카이스퀘어 적합성 검정)
from scipy import stats
chi2, p = stats.chisquare(Obs ,E)
print (" Chi-squared test for given probabilities", "\n",
       "Chi-Squared :", round(chi2, 3), "\n",
       "df :", df,"\n",
       "P-Value :", round(p, 3))
#%% chi-squared independence test(카이제곱 독립성 검정: 결함유형)

import pandas as pd
import numpy as np

defective = np.array([[11,25,27],[15,31,28],[44,24,52],[10,17,16]])
column_names = ['Line1', 'Line2', 'Line3']
row_names    = ['상판', '다리', '서랍', '도색']
table = pd.DataFrame(defective, columns=column_names, index=row_names)

# chi-squared independence test
from scipy import stats
chi22, p2, dof, expected = stats.chi2_contingency(defective)
print (" Pearson's Chi-squared test","\n",
       "Chi-Squared :",round(chi22, 3),"\n",
       "df :",dof,"\n",
       "P-Value :", round(p2, 3))
#%%
"""Statistics using Python by Prof. Uk Jung
8. 분산분석 (Analysis of Variance; ANOVA)
"""
#%% 일원분산분석: 자동차 헤드라이트

# data input
import numpy as np
import pandas as pd
y = np.array([15, 11, 12, 13, 12,
              18, 17, 16, 17, 16,
              22, 23, 19, 18, 19])
group = np.repeat(['A','B','C'],[5,5,5])
group_data = pd.DataFrame({'y':y,'treat':group})
group_data

# multiple boxplots
import seaborn as sns
sns.set(style="whitegrid")
ax = sns.boxplot(x="treat", y="y", data=group_data)
#%% 자동차 헤드라이트 ANOVA

from statsmodels.formula.api import ols
import statsmodels.api as sm

lmFit = ols('y ~ treat', data = group_data).fit()
result=sm.stats.anova_lm(lmFit)
print(result)
#%%
"""Statistics using Python by Prof. Uk Jung
9-1. 상관분석(Correlation Analysis)
"""
#%% 상관분석: twins
# twins.csv 
import pandas as pd
twins=pd.read_csv('./data/twins.csv')

# scatter plot
import matplotlib.pylab as plt
plt.scatter(twins.Foster, twins.Biological, label = "twins data")
plt.legend(loc = "best"); plt.xlabel('Foster'); plt.ylabel('Biological')
plt.show()

#
# Pearson's correlations analysis
import scipy.stats as stats
corr = stats.pearsonr(twins.Foster, twins.Biological)
print (" Pearson's Correlation Analysis","\n",
       "Point estimate of correlation coefficient :",round(corr[0],4),"\n",
       "p-value :", round(corr[1], 4))
#%%
"""Statistics using Python by Prof. Uk Jung
9-2. 회귀분석 (Regression Analysis)
"""
#%%
"""Statistics using Python by Prof. Uk Jung
9-2-1. 단순회귀분석 (Simple Regression Analysis)
"""
#%% 단순선형회귀모형 : cars
import pandas as pd
# cars.csv 
cars=pd.read_csv('./data/cars.csv')
cars.head()
cars_1=cars.drop(['Unnamed: 0'], axis=1)
cars_1

# scatter plot for cars.csv
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(cars_1['speed'], cars_1['dist'], c='black')
plt.xlabel("speed"), plt.ylabel("distance")
plt.show()
#%% model fitting for cars.csv
import statsmodels.formula.api as smf
slr = smf.ols('dist ~ speed', data = cars_1).fit()
dir(slr)
slr.params
print("Intercept: ", round(slr.params[0],3), "\n",
      "speed: ", round(slr.params[1],3))
#%% model graph for cars.csv

abline_values = [slr.params[0] + slr.params[1]*i for i in cars_1['speed']]
plt.plot(cars_1['speed'],cars_1['dist'], 'o')
plt.plot(cars_1['speed'], abline_values, 'b')
plt.xlabel('speed')
plt.ylabel('distance')
#%% Residuals & Fitted value for cars.csv

# Residuals
slr.resid.head()
#slr.resid
#%% Fitted value
slr.fittedvalues.head()
#slr.predict(cars_1['speed'])

#%% Prediction for cars.csv
slr.predict({'speed': [2, 5, 3]})
#%% 단순선형회귀모형: 유의성 검정 및 적합도 for cars.csv
slr.summary()
#%% 단순선형회귀모형: 잔차분석 for cars.csv
# goodness of fit plot, residual plot, normal probability plot of residual
import matplotlib.pyplot as plt
from scipy import stats
import pylab

# residual plot
plt.subplot(1,2,1)
plt.plot(cars_1['speed'], slr.resid, 'o')
plt.title('Residual plot' ); plt.xlabel('speed'); plt.ylabel('Residual')
plt.tight_layout()

# normal probability plot of residual
plt.subplot(1,2,2)
stats.probplot(slr.resid, dist = 'norm', plot = pylab)
plt.title('QQ plot'); plt.xlabel('Theoretical Quantiles'); 
plt.ylabel('Sample Quantiles')
plt.tight_layout()
#%%
"""Statistics using Python by Prof. Uk Jung
9-2-2. 다중회귀분석 (Multiple Regression Analysis)
"""
#%% 다중선형회귀분석 for drywall.csv 
import pandas as pd
# drywall.csv 
drywall=pd.read_csv('./data/drywall.csv')
drywall.head()
#%% scatter plot for drywall.csv 
import seaborn as sns
sns.pairplot(drywall[['Sales', 'Permits', 'Mortgage',
                      'AVacancy', 'OVacancy']], height=1.0)
#%%  model fitting for drywall.csv
import statsmodels.formula.api as smf
mlr = smf.ols('Sales ~ Permits + Mortgage + AVacancy + OVacancy ', 
              data = drywall).fit()
#dir(mlr)
mlr.params
#%% Prediction for drywall.csv
mlr.predict({'Permits': [70, 65],
             'Mortgage': [6, 8],
             'AVacancy': [4, 6],
             'OVacancy': [12, 11]})
#%% t, F, R square  for drywall.csv 
mlr.summary()
#%%
"""Statistics using Python by Prof. Uk Jung
다중회귀분석 연습
"""
#%% swiss data
import pandas as pd

# swiss.csv 
swiss=pd.read_csv('./data/swiss.csv')
swiss.head()
swiss1 = swiss.drop(['Unnamed: 0'], axis=1)
swiss1.head()
#%% scatter plot for swiss.csv 
import seaborn as sns
sns.pairplot(swiss1[['Fertility', 'Agri', 
                      'Exam', 'Edu', 
                      'Cath', 'Infan']], height=1.0)
#%% model fitting for swiss.csv 
   
import statsmodels.formula.api as smf
mlr0 = smf.ols('Fertility ~ Agri + Exam + Edu + Cath + Infan', 
              data = swiss1).fit()
dir(mlr0)
mlr0.params    
#%% t, F, R square for swiss.csv 
mlr0.summary()
#%% residual analysis for swiss.csv 

import matplotlib.pyplot as plt
from scipy import stats
import pylab

mlr0.resid
mlr0.fittedvalues

# residual plot
plt.subplot(1,2,1)
plt.plot(mlr0.fittedvalues, mlr0.resid, 'o')
plt.title('Residual plot' ); plt.xlabel('Fitted value'); plt.ylabel('Residual')
plt.tight_layout()

# normal probability plot of residual
plt.subplot(1,2,2)
stats.probplot(mlr0.resid, dist = 'norm', plot = pylab)
plt.title('QQ plot'); plt.xlabel('Theoretical Quantiles'); 
plt.ylabel('Sample Quantiles')
plt.tight_layout()
#%% Prediction for swiss.csv 
mlr0.predict({'Agri': [54.10, 65.65],
              'Exam': [12.00, 30.10],
              'Edu': [12.00, 5.40],
              'Cath': [50.00, 70.50],
              'Infan': [21.30, 19.90]})
#%% model fitting 1/2/3 for swiss.csv 
      
import statsmodels.formula.api as smf
mlr1 = smf.ols('Fertility ~ Agri + Exam + Cath + Infan', 
              data = swiss1).fit()
mlr2 = smf.ols('Fertility ~ Exam + Edu + Infan', 
              data = swiss1).fit()
mlr3 = smf.ols('Fertility ~ Edu + Cath + Infan', 
              data = swiss1).fit()
print("Adj-Rsqaure for Model 1: ", mlr1.rsquared_adj, "\n",
      "Adj-Rsqaure for Model 2: ", mlr2.rsquared_adj, "\n",
      "Adj-Rsqaure for Model 3: ", mlr3.rsquared_adj)
#%%
"""Statistics using Python by Prof. Uk Jung
9-2-2. 다중회귀분석 (Multiple Regression Analysis)
        - 다중공선성 (Multicollinearity)
"""
#%% selling.csv with multicollinearity
import pandas as pd
selling=pd.read_csv('./data/selling.csv')
print(selling.head())
#%% model fitting for selling.csv 
import statsmodels.formula.api as smf
mlr_selling = smf.ols('price ~ bedrooms + hsize + lotsize', 
              data = selling).fit()
print(mlr_selling.summary())
#%% scatter plot for selling.csv 
import seaborn as sns
sns.pairplot(selling[['price', 'bedrooms', 
                      'hsize', 'lotsize']], height=1.0)
#%% Correlation matrix  for selling.csv 
corrMatrix = selling.corr()
print (corrMatrix)
#%% VIF for  for selling.csv 

from patsy import dmatrices
import statsmodels.api as sm;
from statsmodels.stats.outliers_influence import variance_inflation_factor

y, X = dmatrices('price ~ bedrooms + hsize + lotsize', 
                 selling, return_type = 'dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns 
vif
#%%##################################################################
################# <The End for Statistics>  #########################
#####################################################################
