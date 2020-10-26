# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:20:47 2020

@author: shyoung.kim
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
# import math

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
member_IDs= ['m_01', 'm_02', 'm_03', 'm_04']
before_ex = [27, 35, 40, 33]
after_ex = [30, 38, 42, 37]

colors = ['r', 'g', 'b', 'm']
bar_width= 0.4      # defualt value : 0.8
bar_height = 0.2

member_size = len(member_IDs)
plt.bar(np.arange(member_size), before_ex, color = colors, tick_label = member_IDs, width = bar_width)
plt.barh(np.arange(member_size), before_ex, color = colors, tick_label = member_IDs, height = bar_height)

#%%
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


#%%
