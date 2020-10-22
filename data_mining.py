import os  # 디렉토리 설정
import numpy as np  # 선형대수, 행렬, 벡터
import pandas as pd  # CSV파일 읽기, DataFrame 객체, 평균, 중앙값, 분산, 표준편차, 사분위수, 상관관계
import matplotlib.pyplot as plt  # 박스플랏, 산점도
import scipy.stats as stats  # 정규분포, t분포, 신뢰구간(z분포, t분포), 가설검정
import statsmodels.api as sm # 비율의 신뢰구간, 비율의 가설검정, two-sample ttest, 평균차의 신뢰구간, 회귀분석
import statsmodels.formula.api as smf
pd.set_option('display.max_columns', None) # show all columns when displaying dataset

# 디렉토리 설정
os.getcwd()
os.chdir('C:/Data/')
os.getcwd()

#%%
##############################
# Regression - Usedcar with categorical
##############################
usedcar2 = pd.read_csv('usedcar2.csv')
usedcar2
usedcar2.describe()
usedcar2['Color'].value_counts()

# 색상별 가격차
usedcar2.groupby('Color').mean()

# 더미변수 생성 #
usedcar2 = pd.get_dummies(usedcar2, columns=['Color'],prefix='I',drop_first=True)
usedcar2

# 다중회귀분석
usedcar2.lm = smf.ols('Price ~ Odometer+I_white+I_silver', data=usedcar2).fit()
usedcar2.lm.summary()

# 색상별 가격차, Odometer=35000 일때
temp=[[35000,1,0],[35000,0,1],[35000,0,0]]
temp=pd.DataFrame(temp,columns=['Odometer','I_white','I_silver'])
usedcar2.lm.predict(temp)

# 잔차분석
yhat = usedcar2.lm.fittedvalues
residual = usedcar2.lm.resid
# normality
stats.probplot(residual, dist="norm", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()
# equal variance
plt.scatter(yhat, residual)
plt.show()

#%%
##############################
# 로지스틱회귀분석 - directmail
##############################
directmail = pd.read_csv('directmail.csv')
directmail

# 결측치 제거
directmail = directmail.dropna()
len(directmail)

# 더미변수 생성 #
directmail = pd.get_dummies(directmail, columns=['GENDER'], drop_first=True)
directmail.head(8)

# Logistic Regression #
full_m = smf.glm('RESPOND ~ AGE+BUY18+CLIMATE+FICO+INCOME+MARRIED+OWNHOME+GENDER_M',
                 data=directmail, family=sm.families.Binomial()).fit()
full_m.summary()

# Backward Elimination
back1_m = smf.glm('RESPOND ~ AGE+BUY18+CLIMATE+FICO+MARRIED+OWNHOME+GENDER_M',
                 data=directmail, family=sm.families.Binomial()).fit()
back1_m.summary()

# 골라낸 변수들로 final 돌리기 #
final_m = smf.glm('RESPOND ~ AGE+BUY18+CLIMATE+FICO+MARRIED+OWNHOME',
                  data=directmail, family=sm.families.Binomial()).fit()
final_m.summary()

# Prediction
smith = [35, 1, 15, 800, 50,1, 0, 1] 
johnson = [36, 0, 19, 900, 55, 0, 1, 1]
people = [smith, johnson] # to make a dataframe
people = pd.DataFrame(people,columns=['AGE','BUY18','CLIMATE','FICO','INCOME', 'MARRIED','OWNHOME','GENDER_M'])
final_m.predict(people)
y_actual = directmail['RESPOND']
y_actual.value_counts()
y_actual.value_counts() / len(directmail)

# odds ratio #
np.exp(final_m.params)

#%%
##############################
# K-nearest neighbor - directmail
from sklearn.neighbors import KNeighborsClassifier
##############################
directmail = pd.read_csv('directmail.csv')
directmail = directmail.dropna()
directmail = pd.get_dummies(directmail, columns=['GENDER'], drop_first=True)
X = directmail[['AGE','BUY18','CLIMATE','FICO','INCOME', 'MARRIED','OWNHOME','GENDER_M']]
Y = directmail['RESPOND']

#Create a KNN Classifier
knn = KNeighborsClassifier(n_neighbors=100, metric='euclidean')
knn.fit(X, Y)

# Prediction
smith = [35, 1, 15, 800, 50,1, 0, 1] 
johnson = [36, 0, 19, 900, 55, 0, 1, 1]
people = [smith, johnson] # to make a dataframe
people = pd.DataFrame(people,columns=['AGE','BUY18','CLIMATE','FICO','INCOME', 'MARRIED','OWNHOME','GENDER_M'])
knn.predict_proba(people)

#%%
##############################
# Naive Bayes - directmail
from sklearn.naive_bayes import GaussianNB
##############################
directmail = pd.read_csv('directmail.csv')
directmail = directmail.dropna()
directmail = pd.get_dummies(directmail, columns=['GENDER'], drop_first=True)
X = directmail[['AGE','BUY18','CLIMATE','FICO','INCOME', 'MARRIED','OWNHOME','GENDER_M']]
Y = directmail['RESPOND']

#Create a Gaussian NB Classifier
naivebayes = GaussianNB()

# Train the model using the training sets
naivebayes.fit(X,Y)

# Prediction
smith = [35, 1, 15, 800, 50,1, 0, 1] 
johnson = [36, 0, 19, 900, 55, 0, 1, 1]
people = [smith, johnson] # to make a dataframe
people = pd.DataFrame(people,columns=['AGE','BUY18','CLIMATE','FICO','INCOME', 'MARRIED','OWNHOME','GENDER_M'])
naivebayes.predict_proba(people) 

#%%
#####################################################################################
# Model Selection #
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
##############################
directmail = pd.read_csv('directmail.csv')
directmail = directmail.dropna()
directmail = pd.get_dummies(directmail, columns=['GENDER'], drop_first=True)
X = directmail[['AGE','BUY18','CLIMATE','FICO','INCOME', 'MARRIED','OWNHOME','GENDER_M']]
Y = directmail['RESPOND']

# training vs test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
len(X_train)
len(X_test)
traindat = pd.DataFrame(y_train,columns=['RESPOND']).join(X_train)

# Four competing models
model1 = smf.glm('RESPOND ~ AGE+BUY18+CLIMATE+FICO+MARRIED+OWNHOME',
                  data=traindat, family=sm.families.Binomial()).fit()
model2 = smf.glm('RESPOND ~ AGE+BUY18+CLIMATE+FICO+MARRIED+OWNHOME+INCOME+GENDER_M',
                  data=traindat, family=sm.families.Binomial()).fit()
model3 = KNeighborsClassifier(n_neighbors=100, metric='euclidean')
model3.fit(X_train, y_train)
model4 = GaussianNB()
model4.fit(X_train, y_train)

# Accuracy at cutoff=0.075
# model 1: logistic
y_prob1 = model1.predict(X_test)
y_pred1=np.zeros(len(X_test), dtype=int)
y_pred1[y_prob1>0.075]=1
tab1=pd.crosstab(y_test, y_pred1)
tab1
np.trace(tab1)/len(y_test)
# model 2: logistic
y_prob2 = model2.predict(X_test)
y_pred2=np.zeros(len(X_test), dtype=int)
y_pred2[y_prob2>0.075]=1
tab2=pd.crosstab(y_test, y_pred2)
tab2
np.trace(tab2)/len(y_test)
# model 3: KNN
y_prob3 = model3.predict_proba(X_test)[:,1]
y_pred3=np.zeros(len(X_test), dtype=int)
y_pred3[y_prob3>0.075]=1
tab3=pd.crosstab(y_test, y_pred3)
tab3
np.trace(tab3)/len(y_test)
# model 4: Naive Bayes
y_prob4 = model4.predict_proba(X_test)[:,1]
y_pred4=np.zeros(len(X_test), dtype=int)
y_pred4[y_prob4>0.075]=1
tab4=pd.crosstab(y_test, y_pred4)
tab4
np.trace(tab4)/len(y_test)

# ROC Curve #
# sensitivity & specificity
fpr1, tpr1, thresholds1 = roc_curve(y_test, y_prob1)
fpr2, tpr2, thresholds2 = roc_curve(y_test, y_prob2)
fpr3, tpr3, thresholds3 = roc_curve(y_test, y_prob3)
fpr4, tpr4, thresholds4 = roc_curve(y_test, y_prob4)
# AUROC
logit_roc_auc1 = roc_auc_score(y_test, y_prob1 )
logit_roc_auc2 = roc_auc_score(y_test, y_prob2 )
logit_roc_auc3 = roc_auc_score(y_test, y_prob3 )
logit_roc_auc4 = roc_auc_score(y_test, y_prob4 )
# Curve plotting
plt.figure()
plt.plot(fpr1, tpr1, label='Logistic Regression 1(area = %0.2f)' % logit_roc_auc1)
plt.plot(fpr2, tpr2, label='Logistic Regression 2(area = %0.2f)' % logit_roc_auc2)
plt.plot(fpr3, tpr3, label='KNN(area = %0.2f)' % logit_roc_auc3)
plt.plot(fpr4, tpr4, label='Naive Bayes(area = %0.2f)' % logit_roc_auc4)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#%%
#####################################################################################
# Decistion Trees #
##############################
from sklearn import tree
import pydotplus
from IPython.display import Image
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from sklearn.tree import export_graphviz
##############################
# hmeq data
hmeq = pd.read_csv('hmeq.txt',sep='\t')
hmeq.dtypes
hmeq.describe()
hmeq
hmeq = hmeq.drop(['REASON','JOB'], axis=1)
# 결측치 제거 혹은 채워넣기
#hmeq = hmeq.dropna()
hmeq = hmeq.fillna(hmeq.median())
#hmeq = hmeq.fillna(value=999)
hmeq.describe()

modelfit_X = hmeq.iloc[:,1:]
modelfit_y = hmeq.iloc[:,0]
modelfit_X
modelfit_y
X_train, X_test, y_train, y_test = train_test_split(modelfit_X, modelfit_y, 
                                                test_size=0.4, random_state=0)
len(X_train)
len(X_test)
Xname = ['LOAN','MORTDUE','VALUE','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']
yname = ['good','bad']

# default tree
cart0 = tree.DecisionTreeClassifier(criterion='gini',random_state=0)
cart0.fit(X_train, y_train)
dot_data = export_graphviz(cart0, out_file=None, feature_names=Xname,
    class_names=yname,filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.set_size('"10,10"') 
Image(graph.create_png())

# simple tree
cart1 = tree.DecisionTreeClassifier(criterion='gini',max_depth=2,random_state=0)
cart1.fit(X_train, y_train)
dot_data = export_graphviz(cart1, out_file=None, feature_names=Xname,
    class_names=yname,filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# longer tree2
cart2 = tree.DecisionTreeClassifier(criterion='gini',min_impurity_decrease=0.01,min_samples_split=20,random_state=0)
cart2.fit(X_train, y_train)
dot_data = export_graphviz(cart2, out_file=None, feature_names=Xname,
    class_names=yname,filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# longer tree3
cart3 = tree.DecisionTreeClassifier(criterion='gini',min_impurity_decrease=0.005,min_samples_split=20,random_state=0)
cart3.fit(X_train, y_train)
dot_data = export_graphviz(cart3, out_file=None, feature_names=Xname,
    class_names=yname,filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# longer tree4
cart4 = tree.DecisionTreeClassifier(criterion='gini',min_impurity_decrease=0.001,min_samples_split=20,random_state=0)
cart4.fit(X_train, y_train)
dot_data = export_graphviz(cart4, out_file=None, feature_names=Xname,
    class_names=yname,filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.set_size('"10,10"')
Image(graph.create_png())
#%%
##############################
# Accuracy comparison of several trees
y_prob1 = cart1.predict_proba(X_test)[:,1]
y_prob2 = cart2.predict_proba(X_test)[:,1]
y_prob3 = cart3.predict_proba(X_test)[:,1]
y_prob4 = cart4.predict_proba(X_test)[:,1]
y_prob0 = cart0.predict_proba(X_test)[:,1]
#%%
##############################
# ROC Curve #
fpr1, tpr1, thresholds1 = roc_curve(y_test, y_prob1)
fpr2, tpr2, thresholds2 = roc_curve(y_test, y_prob2)
fpr3, tpr3, thresholds3 = roc_curve(y_test, y_prob3)
fpr4, tpr4, thresholds4 = roc_curve(y_test, y_prob4)
fpr0, tpr0, thresholds0 = roc_curve(y_test, y_prob0)
roc_auc1 = roc_auc_score(y_test, y_prob1 )
roc_auc2 = roc_auc_score(y_test, y_prob2 )
roc_auc3 = roc_auc_score(y_test, y_prob3 )
roc_auc4 = roc_auc_score(y_test, y_prob4 )
roc_auc0 = roc_auc_score(y_test, y_prob0 )
plt.figure()
plt.plot(fpr1, tpr1, label='Tree 1(area = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, label='Tree 2(area = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3, label='Tree 3(area = %0.2f)' % roc_auc3)
plt.plot(fpr4, tpr4, label='Tree 4(area = %0.2f)' % roc_auc4)
plt.plot(fpr0, tpr0, label='Tree 0(area = %0.2f)' % roc_auc0)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


#%%
#####################################################################################
# clustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
##############################
# Hierarchical clustering
# USArrests data
usarrest= pd.read_csv("usarrest.csv")
Xname = ['Murder','Assault','UrbanPop','Rape']
Xdata = StandardScaler().fit_transform(usarrest[Xname])
Xdata = pd.DataFrame(Xdata)
Xdata.columns=Xname
Xdata.describe()
# Calculate the linkage: mergings
mergings = linkage(Xdata,method='average')
# Plot the dendrogram, using varieties as labels
plt.figure(figsize=(20,10))
dendrogram(mergings,
           leaf_rotation=90,
           leaf_font_size=20,
           labels = usarrest['State'].values
)
plt.show()
# Calculate means for each cluster
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='average')  
group = cluster.fit_predict(Xdata)  
group = pd.DataFrame(group)
group.columns=['labels']
usarrest = usarrest.join(group)
usarrest
usarrest.groupby('labels').mean()
#%%
##############################
# K-means clustering
# USArrests data
usarrest= pd.read_csv("usarrest.csv")
Xname = ['Murder','Assault','UrbanPop','Rape']
Xdata = StandardScaler().fit_transform(usarrest[Xname])
Xdata = pd.DataFrame(Xdata)
Xdata.columns=Xname
Xdata.describe()
kmean = KMeans(n_clusters=5,algorithm='auto', random_state=1234)
kmean.fit(Xdata)
clus = pd.DataFrame(kmean.predict(Xdata))
clus.columns=['predict']
usarrest = usarrest.join(clus)
usarrest
usarrest.groupby('predict').mean()


#%%
#####################################################################################
# Association Rule
# Install with pip "pip install mlxtend"
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
te = TransactionEncoder()
##############################
#%%
##############################
# Association Rule : example 1
df = pd.read_csv("grocery.csv")
df.head()
# make boolean data
list_data = df.T.apply(lambda x: x.dropna().tolist()).tolist()
te_data = te.fit(list_data).transform(list_data)
grocery_data = pd.DataFrame(te_data, columns=te.columns_)
grocery_data.head()
# association rule
frequent_itemsets = apriori(grocery_data, min_support=0.05, use_colnames=True)
frequent_itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules = rules.sort_values("lift", ascending=False)
rules
# subset of the rules
rules[rules['antecedents'] == {'whole milk'}]
#%%
##############################
# Association Rule : example 2
basket_data = pd.read_csv("marketbasket.csv")
basket_data.head()
# association rule
frequent_itemsets = apriori(basket_data, min_support=0.05, use_colnames=True)
frequent_itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=3)
rules = rules.sort_values("lift", ascending=False)
rules
# subset of the rules
rules[(rules['support']>=0.05) & (rules['confidence']>0.4) & (rules['lift']>4) ]
rules[rules['antecedents'] == {' White Bread'}]
#%%
##############################
# Association Rule : example 3
build_data = pd.read_csv("building.csv",encoding="EUC-KR")
build_data = build_data.drop('building',axis=1) # delete the first column
build_data
# association rule
frequent_itemsets = apriori(build_data, min_support=0.2, use_colnames=True)
frequent_itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=3)
rules = rules.sort_values("lift", ascending=False)
rules
# subset of the rules
rules[rules['antecedents'] == {'학원'}]
#%%
##############################
# Association Rule : example 4
df = pd.read_csv("movie.csv")
df.head(10)
df1 = df.pivot(index='ID', columns='Item', values='Item')
df1.head()
# make boolean data
list_data = df1.T.apply(lambda x: x.dropna().tolist()).tolist()
te_data = te.fit(list_data).transform(list_data)
movie_data = pd.DataFrame(te_data, columns=te.columns_)
movie_data.head()
# association rule
frequent_itemsets = apriori(movie_data, min_support=0.2, use_colnames=True)
frequent_itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules = rules.sort_values("lift", ascending=False)
rules

