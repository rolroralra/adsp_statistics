# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:20:47 2020

@author: shyoung.kim
"""
#%%
import os
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import pandas as pd
# import math

os.getcwd()
os.chdir('../data')
os.getcwd()
#%%
data = [19, 10 , 14, 25, 20]
#data.sort(reverse=True)
data.sort()
print(data)
plt.plot(data)
plt.show()

x = np.arange(-4.5, 5, 0.5)
print(x)
y = 2*(x**2)
print(y)
d2_points=list(zip(x,y))
print(d2_points)

plt.plot(x,y)
plt.show()
#%%
x = np.arange(-4.5, 5, 0.5)
print(x)

y1 = 2 * x ** 2
print(y1)

y2 = 5 * x + 30
print(y2)

y3 = 4 * x ** 2 + 10
print(y3)

plt.close('all')

plt.plot(x, y1, x, y2, x, y3)
plt.show()

figure_1 = plt.figure()
plt.plot(x, y1)

figure_2 = plt.figure()
plt.plot(x, y2)

plt.show()

#%%
x = np.round(np.arange(-5, 5, 0.1), 1)
print(x)

y1 = x ** 2 - 2
print(y1)

y2 = 20 * np.cos(x)
print(y2)

plt.close('all')

plt.figure(1)
plt.plot(x, y1)

plt.figure(2)
plt.plot(x, y2)

plt.figure(1)
plt.plot(x, y2)

plt.figure(2)
plt.clf()
plt.plot(x, y1)

plt.show()
#%%
x = np.arange(0, 10, 0.1)
print(x)

y1 = 0.3 * (x - 5) ** 2 + 1
y2 = -1.5 * x + 3
y3 = np.sin(x) ** 2
y4 = 10 * np.exp(-x) + 1

plt.close('all')

row_size=2
col_size=3
plt.subplot(row_size, col_size, 1)
plt.plot(x, y1)
plt.subplot(row_size, col_size, 2)
plt.plot(x, y2)
plt.subplot(row_size, col_size, 4)
plt.plot(x, y3)
plt.subplot(row_size, col_size, 6)
plt.plot(x, y4)

plt.show()
#%%
x = np.linspace(-4, 4, 100)

y1 = x ** 3
y2 = 10 * x ** 2 - 2

plt.plot(x, y1, x, y2)
plt.xlim(-1, 1)
plt.ylim(-3, 3)
plt.show()
#%%
x = np.arange(0, 5, 1)

y1 = x
y2 = x + 1
y3 = x + 2
y4 = x + 3

plt.plot(x, y1, 'm', x, y2, 'y', x, y3, 'k', x, y4, 'c')
plt.plot(x, y1, '-', x, y2, '--', x, y3, ':', x, y4, '-.')
plt.plot(x, y1, 'o', x, y2, '^', x, y3, 's', x, y4, 'd')
plt.plot(x, y1, 'm-o', x, y2, 'x--y', x, y3, 'k:s', x, y4, 'c-d')
#%%
x = np.arange(-4.5, 5, 0.5)
y1 = 2 * x ** 3
y2 = x + 1

plt.plot(x, y1, 'o--r', x, y2, 's-g')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.grid(True)
plt.legend(['korea', 'y2'], loc='best')
plt.text(-3, 100, 'young 1')
plt.text(0, 25, 'old 2')

plt.show()
#%%
###################
# bar
###################
member_IDs= ['m_01', 'm_02', 'm_03', 'm_04']
before_ex = [27, 35, 40, 33]
after_ex = [30, 38, 42, 37]

colors = ['r', 'g', 'b', 'm']
bar_width= 0.4      # defualt value : 0.8
bar_height = 0.2

member_size = len(member_IDs)
plt.bar(np.arange(member_size), before_ex, color = colors, tick_label = member_IDs, width = bar_width)
plt.barh(np.arange(member_size), before_ex, color = colors, tick_label = member_IDs, height = bar_height)

plt.show()
#%%
###################
# bar with two group
###################
member_IDs= ['m_01', 'm_02', 'm_03', 'm_04']
before_ex = [27, 35, 40, 33]
after_ex = [30, 38, 42, 37]

index = np.arange(len(member_IDs))
bar_width = 0.4


plt.bar(index,             before_ex, color='c', align='edge', width=bar_width, label='before')
plt.bar(index + bar_width, after_ex,  color='m', align='edge', width=bar_width, label='after')
plt.xticks(index + bar_width, member_IDs)
plt.legend()
plt.xlabel('ID')
plt.ylabel('count')
plt.title('Before & After')

plt.show()
#%%
###################
# pie
###################
fruit = ['apple', 'banana', 'strawberry', 'orange', 'grape']
result = np.random.randint(0, 10 + 1, size=len(fruit))
plt.figure(figsize=(5, 5))
plt.pie(result)

plt.figure()
explode_value=(0.5, 0.1, 0, 0, 0)
plt.pie(result, labels=fruit, explode=explode_value, autopct='%.1f%%', startangle=90, counterclock=False, shadow=True)

plt.show()
#%%
###################
# histogram
###################
student_num = 100
math = np.random.randint(0, 100 + 1, size=student_num)
english = np.random.randint(0, 100 + 1, size=student_num)
plt.hist(math)
# plt.hist2d(math, english)
plt.hist(math, bins=10)
plt.xlabel('score')
plt.ylabel('frequency')
plt.title('math score histogram')
plt.grid(True)      # plt.grid()

plt.show()
#%%
###################
# scatter plot
###################
data_size=20
mean_value=173
std_value=5
height = stats.norm.rvs(loc=mean_value, scale=std_value, size=data_size, random_state=np.random.randint(time.time())).round()

mean_value=70
std_value=8
weight = stats.norm.rvs(loc=mean_value, scale=std_value, size=data_size, random_state=np.random.randint(time.time())).round()

plt.scatter(height, weight)
plt.xlabel('Height(m)')
plt.ylabel('Weight(kg)')
plt.title('Scatter Plot for Height & Wieght')
plt.grid()

plt.show()

#%%
###################
# boxplot
###################
member_IDs= ['m_01', 'm_02', 'm_03', 'm_04']
before_ex = [27, 35, 40, 33]
after_ex = [30, 38, 42, 37]
sns.boxplot(x=before_ex)

table1 = list(zip(before_ex,after_ex))
table2 = np.transpose([before_ex, after_ex])

# multiple boxplot
dataFrame = pd.DataFrame(table1, columns=['before_ex', 'after_ex'])
dataFrame
dataFrame.dtypes
dataFrame.columns
dataFrame.describe()

sns.boxplot(data=dataFrame, orient='h')
sns.boxplot(data=dataFrame, orient='h', palette='Set2')

#%%
###################
# statistical measure
# 1. central tendency
# 2. variability
# 3. shape
###################
mean_value=75
std_value=15
sample_size=100
sample = stats.norm.rvs(loc=mean_value, scale=std_value, size=sample_size, random_state=np.random.randint(time.time()))

# central tendency
print("sample:", sample, '\n')
print("mean:", np.mean(sample))
print("median:", np.median(sample))
print()
print("mean:", statistics.mean(sample))
print("median:", statistics.median(sample))
print("mode:", statistics.mode(sample))


# variability
print("variance:", np.var(sample))
print("variance:", np.var(sample, ddof=1))
print("variacne:", statistics.variance(sample))
print()

print("standard deviation:", np.std(sample))
print("standard deviation:", np.std(sample, ddof=1))
print("standard deviation:", statistics.stdev(sample))
print()

print("max:", np.max(sample))
print("min:", np.min(sample))
print("range:", np.max(sample) - np.min(sample))
print()

quartile_1, quartile_3 = np.quantile(sample, [0.25, 0.75])
percentile_25, percentile_75 = np.percentile(sample, [25, 75])
print("IQR(Inter Quartile Range:", quartile_3 - quartile_1)
print()

# shape
from scipy.stats import skew
from scipy.stats import kurtosis
print("skewness:", skew(sample))
print("kurtosis:", kurtosis(sample))
print()

dataFrame = pd.DataFrame(sample)
print(dataFrame.describe())
#%%
