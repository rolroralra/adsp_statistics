# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:08:43 2020

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
import math

from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import expon

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.formula.api import ols

os.getcwd()
os.chdir('../data')
os.getcwd()
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
########################
#
########################
np.random.seed(int(time.time()))

data_size = 100
sample_size = 30
data_range=(0,10)
sample_mean_array=[]
for i in range(data_size):
    sample = np.random.randint(data_range[0], data_range[1], size = sample_size)
    sample_mean_array.append(np.mean(sample))

# sample_mean_array = np.array(sample_mean_array)
print(sample_mean_array)
print()

plt.hist(sample_mean_array, bins=100, range=data_range)
plt.xlabel('sample_mean')
plt.ylabel('frequency')
plt.title('Histogram of sample mean')
plt.show()
#%%
##########################
# confidence interval
##########################
population_mean = 257
sample_data = [260, 265, 250, 270, 272, 258, 262, 268, 270, 252]
sample_size = len(sample_data)
dof = sample_size - 1

confidence_level = 0.95
sample_mean = np.mean(sample_data)
sample_std_deviation = np.std(sample_data, ddof=1)

print(f"population_mean: {population_mean}")
print(f"sample_szie: {sample_size}")
print(f"sample_mean: {sample_mean}")
print(f"sample_standard_deviation: {sample_std_deviation}")
print()

alpha = (1 - confidence_level) / 2
t_alpha = stats.t.ppf(confidence_level + alpha, dof)
print("t_alpha:", t_alpha)

delta = t_alpha * sample_std_deviation / np.sqrt(sample_size)

confidence_interval = (sample_mean - delta, sample_mean + delta)
confidence_interval = np.round(confidence_interval, 3)
print(f"{int(100 * confidence_level)}% confidence interval: {confidence_interval}")
print()

##############################################
# hypothesis test for population mean
##############################################
# way1. rejection region
significance_level=0.05

print("significance_level:", significance_level)
t_alpha = stats.t.ppf(1 - significance_level / 2, dof)
delta = t_alpha * sample_std_deviation / np.sqrt(sample_size)
rejection_region = np.round((population_mean - delta, population_mean + delta), 3)
print(f"rejection region with significance_level {int(significance_level * 100)}%: {rejection_region}")
print()

# way2. p-value
t_sample_value = (sample_mean - population_mean) / sample_std_deviation * np.sqrt(sample_size)
p_value = round(2 * stats.t.cdf(-abs(t_sample_value), dof), 3)
print(f"t_statatistic_value: {t_sample_value}")
print(f"p-value for mean != {population_mean}: {p_value}")
print()

# by using stats.ttest_1samp(sample, population_mean)
print(stats.ttest_1samp(sample_data, population_mean))
print()

t_sample_value = (sample_mean - population_mean) / sample_std_deviation * np.sqrt(sample_size)
p_value_left = round(stats.t.cdf(t_sample_value, dof), 3)
print(f"t_statatistic_value: {t_sample_value}")
print(f"p-value for mean < {population_mean}: {p_value_left}")
print()

p_value_right = round(1 - stats.t.cdf(t_sample_value, dof), 3)
print(f"t_statatistic_value: {t_sample_value}")
print(f"p-value for mean > {population_mean}: {p_value_right}")
print()
#%%
##############################################
# hypothesis test for population variance
# (n - 1) * S**2 / variance  ~ chi2(n - 1) 
##############################################
population_stddev=5
sample_data = [198, 201, 199, 189, 200, 199, 198, 189, 205, 195]
sample_size = len(sample_data)
dof = sample_size - 1
significance_level = 0.05

print(f"population standard deviation: {population_stddev}")
print(f"sample_size: {sample_size}")
print()

sample_mean = np.mean(sample_data)
sample_variance = np.var(sample_data, ddof=1)
sample_stddev = np.std(sample_data, ddof=1)
print(f"sample mean : {sample_mean}")
print(f"sample variance: {sample_variance}")
print(f"sample standard deviation: {sample_stddev}")
print()


chi_alpha = stats.chi2.ppf(1 - significance_level, dof)
critical_value = chi_alpha * sample_variance / dof
print(f"critical value: {critical_value}")
print(f"rejection region for variance > {population_stddev ** 2} : [{critical_value},]")
print()

chi_value = sample_variance * dof / population_stddev ** 2
print("chi_value", chi_value)
p_value = round(1 - stats.chi2.cdf(chi_value, dof), 3)
print(f"p_value for variance > {population_stddev ** 2}: {p_value}")
print(f"Hypothesis test for variance > {population_stddev ** 2}, it can't be rejected")
print()
#%%
######################################################################
# hypothesis test for two independent population means difference
# using t distribution
# stats.ttest_ind(sample1, sample2, eqaul_var=True)
######################################################################
whours = pd.read_csv('whours.csv')
sample1 = whours[whours['gender'] == 'M'].hours
sample2 = whours[whours['gender'] == 'F'].hours


tstatistic, pvalue = stats.ttest_ind(sample1, sample2, equal_var=True)
print(f"tstatistic: {round(tstatistic, 3)}, p-value: {round(pvalue, 3)}")
tstatistic, pvalue = stats.ttest_ind(sample1, sample2, equal_var=False)
print(f"tstatistic: {round(tstatistic, 3)}, p-value: {round(pvalue , 3)}")
print()
#%%
######################################################################
# hypothesis test for two related population means difference (paired sample)
# using t distribution
# stats.ttest_rel(sample1, sample2)
######################################################################
data = pd.read_csv('salespairs.csv')
sample1 = data[data['status'] == 1].sales
sample2 = data[data['status'] == 2].sales
diff_array = sample2.values - sample1.values

tstatistic, pvalue = stats.ttest_rel(sample2, sample1)  # sample2 - sample1
print(f"tstatistic: {round(tstatistic, 3)}, p-value: {round(pvalue, 3)}")
print()

#%%
######################################################################
# hypothesis test for goodness of fit for categorical data (chi-squared 적합성 검정)
# using chi2 distribution
# stats.chisquare(data, probability)
######################################################################
observed_data = np.array([285, 66, 55, 44])
ratios = np.array([0.68, 0.13, 0.11, 0.08])
total_num = observed_data.sum()

expected_values = total_num * ratios

total_category_count = len(observed_data)
dof = total_category_count - 1

statistic, pvalue = stats.chisquare(observed_data, expected_values)
print(f"chi-statistic: {statistic}, p-value: {pvalue}")
print()
#%%
######################################################################
# hypothesis test for independence for categorical data (chi-squared 독립성 검정)
# using chi2 distribution
# stats.chi2_contigency(dataFrame)
######################################################################
data_table = np.array([[11, 25, 27], [15, 31, 28], [44, 24, 52], [10, 17, 16]])
column_names = ['Line1', 'Line2', 'Line3']
row_names = ['Table', 'Leg', 'Case', 'Color']

dataFrame = pd.DataFrame(data_table, columns=column_names, index=row_names)
print(dataFrame)
print(dataFrame.describe())
print()

chi22, p2, dof, expected = stats.chi2_contingency(dataFrame)
print(f"chi2-statistic: {chi22}")
print(f"p-value: {p2}")
print(f"degree of freedom: {dof}")
print(f"expected_values:\n{expected}")
print()
#%%
######################################################################
# ANOVA (Analysis of Variance)
#
# factor
# treatment
######################################################################
# One Way ANOVA
######################################################################
k = treatment_sample_size = 5
y = [15, 11, 12, 13, 12,
     18, 17, 16, 17, 16,
     22, 23, 19, 18, 19]
# treatments = np.repeat(['A', 'B', 'C'], [5, 5, 5])
treatments = ['A'] * 5 + ['B'] * 5 + ['C'] * 5
data = pd.DataFrame({'y': y, 'treatment': treatments})
print(data)
print()

sns.boxplot(x='treatment', y='y', data=data, orient='v')
# sns.boxplot(x='y', y='treatment', data=data, orient='h')

# Ordinary Least Square
from statsmodels.formula.api import ols
import statsmodels.api as sm

fitted_linear_model = ols(f'{data.columns[0]} ~ {data.columns[1]}', data = data).fit()
anova_table = sm.stats.anova_lm(fitted_linear_model)
print(anova_table)
print()

print("p-value for ANOVA, Hypothesis is that there is no difference of means for each treatment groups")
print(": ", anova_table['PR(>F)']['treatment'])
print()
#%%
######################################################################
# Correlation Analysis
#
# 1. Pearson Correlation Coefficient
# 2. Spearman Correlation Coefficient
# 3. Kendall's Tau
######################################################################
data = pd.read_csv('twins.csv')
data.head()

x = data[data.columns[0]]
y = data[data.columns[1]]
plt.scatter(x, y, label= 'twins data')
plt.xlabel(x.name)
plt.ylabel(y.name)
plt.legend(loc='best')

corr_coef = stats.pearsonr(x, y)
print("Pearson's Correlation Analysis")
print(f"Point estimate of Correlation Coefficient r : {round(corr_coef[0], 4)}")
print(f"p-value : {round(corr_coef[1], 4)}")
print()
#%%
######################################################################
# Regression Analysis
#
# 1. Linear Regression
#       1) simple regression
#       2) multiple regression
#       3) polynomial regression
#
# 2. Non-linear Regression  
######################################################################
data = pd.read_csv('cars.csv')
print(data.head(5))
print()

data = data.drop(data.columns[0], axis=1)
print(data.head(5))
print()

x = data[data.columns[0]]
y = data[data.columns[1]]
plt.scatter(x, y, label='sccater plot')
plt.xlabel(x.name)
plt.ylabel(y.name)
plt.grid()
plt.title('scatter plot')

import statsmodels.formula.api as smf
fitted_lm = smf.ols(f"{y.name} ~ {x.name}", data = data).fit()
dir(fitted_lm)
print(fitted_lm.params)
print()


## params
a = round(fitted_lm.params[0], 3)
b = round(fitted_lm.params[1], 3)
print("simple linear regression by OLS (Ordinary Least Square)")
print(f"y = {a} + {b}x")
print()


## fitted values
# fitted_value = x.apply(lambda x: a + b * x)
fitted_values = fitted_lm.fittedvalues
# fitted_values = fitted_lm.predict(x)
# predicted_value = fitted_lm.predict({"speed":[1,2,3]})

plt.figure()
plt.plot(x, fitted_values, '-r', label='fitted value')
plt.plot(x, y, 'ob', label='observed value')
plt.xlabel(x.name)
plt.ylabel(y.name)
plt.grid()
plt.legend()
plt.title('Linear Regression')


## residuals
residuals = fitted_lm.resid


print(fitted_lm.summary())
print()

# evaluating model significance (t, F)
print("t-values:\n", fitted_lm.tvalues)
print()
print("p-value of t:\n", fitted_lm.pvalues)
print()
print("F-value :", fitted_lm.fvalue)
print()
print("p-value of F:", fitted_lm.f_pvalue)
print()

# evaluating goodness of fit of model (R-squared : coefficient of determination R^2)
print("R-squred :", fitted_lm.rsquared)
print("Adj. R-squared :", fitted_lm.rsquared_adj)
print()

#%%
# Residual Analysis
# 1. Residual Plot 
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(x, residuals, 'o', label='residual')
plt.title('Residual Plot')
plt.xlabel(x.name)
plt.ylabel('Residual')
plt.tight_layout()

# 2. normal probability plot of residual (QQ Plot : Quantile Quantile Plot)
# import pylab   # deprecated
plt.subplot(1, 2, 2)
stats.probplot(residuals, dist='norm', plot = plt)
plt.title('QQ Plot')
plt.ylabel('Sample Quantiles')
plt.tight_layout()
#%%
