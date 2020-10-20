# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:02:07 2020

@author: shyoung.kim
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

bin_cdf = stats.binom.cdf(10, 200, 0.04)
print("P(x<=10) when x~bin(200,0.04):", round(bin_cdf,3))

poi_cdf = stats.poisson.cdf(10, 8)
print("P(x<=10) when x~Poisson(8):", round(poi_cdf,3))


#%%
from scipy.stats import norm
# expected_value=3
# standard_deviation=2
# x=2.5
# print("normal pdf=", round(norm.pdf(x, expected_value, standard_deviation), 3))
# print("normal cdf=", round(norm.cdf(x, expected_value, standard_deviation), 3))
# percentile=0.3
# print("noraml quntile=", round(norm.ppf(percentile, expected_value, standard_deviation), 3))
# print("normal random variables=", norm.rvs(expected_value, standard_deviation, size=5))


# 어느 학교 학생들의 신장이 정규분포를 따르며, 평균신장은 175cm이고
# 표준편차는 3cm이다. Python을 이용하여 다음 물음에 답하여라.

# 1) 신장이 177cm가 넘는 학생들의 비율은 얼마인가?
# 2) 학생의 신장이 하위 20%에 속하기 위해서는 최대한 몇 cm가 되어야 하는가?
expected_value=175.0
standard_deviation=3
print("1)", round(1- norm.cdf(177, expected_value, standard_deviation), 3))
#print("1)", round(1- norm.cdf((177 - expected_value) / standard_deviation), 3))
print("2)", round(norm.ppf(0.2, expected_value, standard_deviation), 3))
print("")

# 다음 물음을 Python을 이용해서 구하여라.
# 1) P[Z < -1.64] + P[Z > 1.64]
# 2) P[|Z| < z] = 0.9인 z의 값을 구하여라
# 3) X~N(3,2^2)에서 P[2 < X < 5]를 구하여라.
print("1)", round(norm.cdf(-1.64) + 1 - norm.cdf(1.64),3))
print("2)", round(norm.ppf(0.05),3))

expected_value=3
standard_deviation=2
print("3)", round(norm.cdf(5,expected_value,standard_deviation) - norm.cdf(2,expected_value,standard_deviation), 3))
#print("3)", round(norm.cdf((5-expected_value)/standard_deviation) - norm.cdf((2-expected_value)/standard_deviation), 3))



#%%
