# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:08:43 2020

@author: shyoung.kim
"""
#%%
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import math
#%%
###################
# covariance -- COV(X,Y)
# correlation coefficient -- CORR(X,Y)
###################
sample_mean = 175
sample_stddev = 5
sample_size = 100
sample = stats.norm.rvs(size=sample_size, loc=sample_mean, scale=sample_stddev, random_state=np.random.randint(time.time()))
sample2 = stats.norm.rvs(size=sample_size, loc=sample_mean, scale=sample_stddev, random_state=np.random.randint(time.time()))

sample2 = sample2 / 2 - 18

# covariance matrix
covariance_matrix = np.cov(sample, sample2)     # ddof = 1 (default)
np.cov([sample, sample2], ddof=1)
np.cov(sample, y=sample2, ddof=1)

print("COV(X,Y) =", covariance_matrix[0][1])
#print("correlaion coefficient(CORR(X,Y)):", covariance_matrix[0][1] / np.sqrt(covariance_matrix[0][0]) / np.sqrt(covariance_matrix[1][1])) 

# correlation coefficient matrix
correlation_coefficient_matrix = np.corrcoef(sample, sample2) # DeprecationWarning: bias and ddof have no effect and are deprecated
print("CORR(X,Y) =", correlation_coefficient_matrix[0][1])
print()
#%%
#############################
# Bernoulli Distribution
# X ~ Ber(p)
#############################
# from scipy.stats import binom
n = 1
p = 0.6
k = 0

#############################
# Bionominal Distribution
# X ~ Bin(n,p)
#############################
from scipy.stats import binom
n = 200
p = 0.04
k = 10
cumulative_prob = binom.cdf(k, n, p)
#cumulative_prob = binom.cdf(k=10, n=200, p=0.04)

print(f"When X ~ Bin({n},{p}),\tP(X <= {k}) = {cumulative_prob}")
print("When X ~ Bin({n},{p}),\tP(X <= {k}) = {cumulative_prob}".format(n= 200, p=0.04, k=10, cumulative_prob=cumulative_prob))
print("When X ~ Bin({},{}),\tP(X <= {}) = {}".format(n, p, k, cumulative_prob))
print("When X ~ Bin({0},{1}),\tP(X <= {2}) = {3}".format(n, p, k, cumulative_prob))
print()
#%%
#############################
# Poisson Distribution
# X ~ Poisson(lamda)   , lamda = n * p
#############################
from scipy.stats import poisson
mean = 8
k = 10

cumulative_prob = poisson.cdf(k, mean)
print(f"When X ~ Poisson({mean}),\tP(X <= {k}) = {cumulative_prob}")
print()
#%%
#############################
# Normal Distribution
# X ~ N(mean, variance)
#############################
from scipy.stats import norm

mean = loc = 3            # loc
std_variance = scale = 2    # scale
x = 2.5

pdf_value = norm.pdf(x, loc, scale)
print(f"When X ~ N({loc}, {scale}^2),\t pdf(X = {x}) = {pdf_value}")
cdf_value = norm.cdf(x, loc, scale)
print(f"When X ~ N({loc}, {scale}^2),\t cdf(X <= {x}) = {cdf_value}")

# ppf: percentage point function (inverse function of cdf)
p = 0.25
ppf_value = norm.ppf(p, loc, scale)
print(f"When X ~ N({loc}, {scale}^2),\t ppf(p = {p}) = {ppf_value}")
print(f"When X ~ N({loc}, {scale}^2),\t IQR = [{norm.ppf(0.25, loc, scale)}, {norm.ppf(0.75, loc, scale)}]")

# rvs : random variates
sample_size = 10
print(f"Random Variates (size :{sample_size}) from X ~ N({loc}, {scale}^2)\n", norm.rvs(loc,scale, size=sample_size))
print()
#%%
#############################
# Gamma Distribution
# X ~ Gamma(k, theta)   but in scipy, theta = 1
# f(x;k, theta) = x**(k-1) * exp(-x / theta) / theta ** k / gamma_function(k)
#############################
from scipy.stats import gamma
k = 1
x = 1

pdf_value = gamma.pdf(x, k)
print(f"When X ~ Gamma({k}, 1),\t pdf(X = {x}) = {pdf_value}")

cdf_value = gamma.cdf(x, k)
print(f"When X ~ Gamma({k}, 1),\t cdf(X <= {x}) = {cdf_value}")

# ppf: percentage point function (inverse function of cdf)
p = 0.25
ppf_value = gamma.ppf(p, k)
print(f"When X ~ Gamma({k}, 1),\t ppf(p = {p}) = {ppf_value}")
print(f"When X ~ Gamma({k}, 1),\t IQR = [{gamma.ppf(0.25, k)}, {gamma.ppf(0.75, k)}]")

# rvs : random variates
sample_size = 10
print(f"Random Variates (size :{sample_size}) from X ~ Gamma({k}, 1)\n", gamma.rvs(k, size=sample_size))
print()
#%%
#############################
# Expotential Distribution
# X ~ Exp(lambd)  ... (X ~ Gamma(1, 1 / lambda))
# f(x;lambda) = lambda * exp(-x * lambda)
# in scipy, scale = 1 / lambda
#############################
from scipy.stats import expon
lamda = 1
x = 1

pdf_value = expon.pdf(x, scale=1/lamda)
print(f"When X ~ Expon({lamda}),\t pdf(X = {x}) = {pdf_value}")

cdf_value = expon.cdf(x, scale=1/lamda)
print(f"When X ~ Expon({lamda}),\t cdf(X <= {x}) = {cdf_value}")

# ppf: percentage point function (inverse function of cdf)
p = 0.25
ppf_value = expon.ppf(p, scale=1/lamda)
print(f"When X ~ Expon({lamda}),\t ppf(p = {p}) = {ppf_value}")
print(f"When X ~ Expon({lamda}),\t IQR = [{gamma.ppf(0.25, k)}, {gamma.ppf(0.75, k)}]")

# rvs : random variates
sample_size = 10
print(f"Random Variates (size :{sample_size}) from X ~ Expon({lamda})\n", expon.rvs(k, size=sample_size))
print()