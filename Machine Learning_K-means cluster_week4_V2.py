# -*- coding: utf-8 -*-
"""
Created on 20 March 2016

@author: Jeff Hsueh
"""

#from pandas import Series, DataFrame
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
 
#Load the dataset
data = pd.read_csv("tree_ool_pds.csv")

#upper-case all DataFrame column names
#data.columns = map(str.upper, data.columns)

# Data Management
data_clean = data.dropna()
recode1 = {1:1, 2:0,3:0}
republican = {1:1, 2:0 , 3:0, 4:0}
democrat = {1:0, 2:1 , 3:0, 4:0}
inde = {1:0, 2:0 , 3:1, 4:0}

#data_clean['have_sex_outside_race']= data_clean['W1_E7'].map(recode1)
data_clean['is_repubilcan']= data_clean['W1_C1'].map(republican)
data_clean['is_democrat']= data_clean['W1_C1'].map(democrat)
data_clean['is_inde']= data_clean['W1_C1'].map(inde)
data_clean['is_arrested']= data_clean['W1_P9'].map(recode1)
data_clean['is_unemployed']= data_clean['W1_P11'].map(recode1)
data_clean['have_child']= data_clean['W1_P17'].map(recode1)
#data_clean['is_optimistc_mariage']= data_clean['W1_E2'].map(recode1)
#data_clean['willing_to_date_out']= data_clean['W1_E4'].map(recode1)

def same_sex_agree (row): # Agree = 1; Disagree = 0
    if row['W1_J2']  == 1:
        return "1"
    elif row['W1_J2']  == 2:
        return "1"
    elif row['W1_J2']  == 3:
        return "0"
    else:
        return np.nan
        
def god_is_anger(row):  # Agree = 1 ; Disagree = 0
    if row['W1_M13_B'] == '1':
        return "1"
    elif row['W1_M13_B'] =='2' :
        return "1"
    elif row['W1_M13_B'] =='3':
        return "0"
    elif row['W1_M13_B'] == '4':
        return "0" 
    else:
        return np.nan

def is_highschool (row): # Agree = 1 ; Disagree = 0
    if row['PPEDUCAT']   == 1 :
        return "0"
    elif row['PPEDUCAT']   == 2:
        return "1"
    elif row['PPEDUCAT']   == 3:
        return "1"
    elif row['PPEDUCAT']   == 4:
        return "1"
    else:
        return np.nan

def is_male (row) : # male = 1 ; female = 0
    if row ['PPGENDER'] == 1 :
        return'1'
    elif row ['PPGENDER'] == 2 :
        return "0"
    else:
        return np.nan
        
def is_white(row):   # white = 1 ; others = 0
    if row['PPETHM']  == 1 :
        return "1"
    else :
        return "0"
def is_black(row):  # black = 1 ; others = 0
    if row["PPETHM"] == 2 :
        return "1"
    else: 
        return "0"
        
def W2_QL3_recode(row):
    if row['W2_QL3'] == '1':
        return 1
    elif row['W2_QL3'] == '2':
        return 2
    elif row['W2_QL3'] == '3':
        return 3
    elif row['W2_QL3'] == '4':
        return 4
    else:
        return np.nan
def W1_M13_B_recode(row):
    if row['W1_M13_B'] == '1':
        return 1
    elif row['W1_M13_B'] == '2':
        return 2
    elif row['W1_M13_B'] == '3':
        return 3
    elif row['W1_M13_B'] == '4':
        return 4
    else:
        return np.nan
    
        
data_clean['same_sex_agree'] = data_clean.apply(lambda row : same_sex_agree(row),axis=1)
#print (data_clean['same_sex_agree'])

#data_clean['god_is_anger'] = data_clean.apply(lambda row : god_is_anger(row),axis=1)
##print (data_clean['god_is_anger'])

data_clean['is_highschool'] = data_clean.apply(lambda row : is_highschool(row),axis=1)
#print (data_clean['is_highschool'])

data_clean['is_male'] = data_clean.apply(lambda row : is_male(row),axis=1)
#print (data_clean['is_male'])

data_clean['is_white'] = data_clean.apply(lambda row : is_white(row),axis=1)
#print (data_clean['is_white'])

data_clean['is_black'] = data_clean.apply(lambda row : is_black(row),axis=1)
#print (data_clean['is_black'])

data_clean['W2_QL3'] = data_clean.apply(lambda row : W2_QL3_recode(row),axis=1)
data_clean['W1_M13_B'] = data_clean.apply(lambda row : W1_M13_B_recode(row),axis=1)

data_clean= data_clean.dropna()

#Split into training and testing sets
cluster = data_clean[['W1_L2_3', 'W1_L2_4', 'W1_L2_5','W1_F4_A', 'W1_M13_B', 
'is_highschool', 'is_male' , 'PPAGECT4', 'PPNET', 'PPINCIMP','W1_P2', 'is_black', 'is_white',
'is_repubilcan', 'is_democrat', 'is_inde','is_arrested', 'is_unemployed',
'have_child', 'W1_P2','W2_QL3']]

cluster.describe()

# standardize cluster to have mean=0 and sd=1
clustervar=cluster.copy()

from sklearn import preprocessing
clustervar['W1_L2_3']=preprocessing.scale(clustervar['W1_L2_3'].astype('float64'))
clustervar['W1_L2_4']=preprocessing.scale(clustervar['W1_L2_4'].astype('float64'))
clustervar['W1_L2_5']=preprocessing.scale(clustervar['W1_L2_5'].astype('float64'))
clustervar['W1_F4_A']=preprocessing.scale(clustervar['W1_F4_A'].astype('float64'))
clustervar['is_highschool']=preprocessing.scale(clustervar['is_highschool'].astype('float64'))
clustervar['is_male']=preprocessing.scale(clustervar['is_male'].astype('float64'))
clustervar['PPAGECT4']=preprocessing.scale(clustervar['PPAGECT4'].astype('float64'))
clustervar['PPNET']=preprocessing.scale(clustervar['PPNET'].astype('float64'))
clustervar['is_black']=preprocessing.scale(clustervar['is_black'].astype('float64'))
clustervar['is_white']=preprocessing.scale(clustervar['is_white'].astype('float64'))
clustervar['is_repubilcan']=preprocessing.scale(clustervar['is_repubilcan'].astype('float64'))
clustervar['is_democrat']=preprocessing.scale(clustervar['is_democrat'].astype('float64'))
clustervar['is_inde']=preprocessing.scale(clustervar['is_inde'].astype('float64'))
clustervar['is_arrested']=preprocessing.scale(clustervar['is_arrested'].astype('float64'))
clustervar['is_unemployed']=preprocessing.scale(clustervar['is_unemployed'].astype('float64'))
clustervar['have_child']=preprocessing.scale(clustervar['have_child'].astype('float64'))

clustervar['W1_M13_B']=preprocessing.scale(clustervar['W1_M13_B'].astype('float64'))
clustervar['PPINCIMP']=preprocessing.scale(clustervar['PPINCIMP'].astype('float64'))
clustervar['W1_P2']=preprocessing.scale(clustervar['W1_P2'].astype('float64'))
clustervar['W2_QL3']=preprocessing.scale(clustervar['W2_QL3'].astype('float64'))

"""
Binary Variables:

W1_L2_3 : [Southern Christian Leadership Conference] Are you a member of any of the
following groups/organizations/movements?

W1_L2_4: [Tea Party Movement] Are you a member of any of the following
groups/organizations/movements?

W1_L2_5: [Occupy Wall Street Movement] Are you a member of any of the following
groups/organizations/movements?

W1_F4_A: [To own a home ] For yourself and people like you, how easy or hard is it to reach
these goals?

PPNET: HH Internet Access

"""
"""
Quantitiatives Varialbes:

W1_M13_B: [God is angered by human sin. ]For believers and non-believers,based on your
personal understanding,what do you think God is like

PPAGECT4: Age - 4 Categories

PPINCIMP: Household Income

W1_P2: People talk about social classes such as the poor, the working class, the middle class,
the upper-middle class, and the upper class. Which of these classes would you say you
belong to?

W2_QL3: Some people believe that homosexuality is immoral. To what extent do you agree or
disagree with this perspective?

"""


# split data into train and test sets
clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

# k-means cluster analysis for 1-9 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clus_train.shape[0])

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')

# Interpret 3 cluster solution
model3=KMeans(n_clusters=4)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)
# plot clusters

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')

#plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()

"""
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
# create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
#clus_train.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslist=list(clus_train['index'])
# create a list of cluster assignments
labels=list(model3.labels_)
# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
newlist
# convert newlist dictionary to a dataframe
newclus=DataFrame.from_dict(newlist, orient='index')
newclus
# rename the cluster assignment column
newclus.columns = ['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data
newclus.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_train=pd.merge(clus_train, newclus, on='index')
merged_train.head(n=100)
# cluster frequencies
merged_train.cluster.value_counts()

"""
END multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""

# FINALLY calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)


# validate clusters in training data by examining cluster differences in How people rate Gays/lesbians using ANOVA
# first have to merge How people rate Gays/lesbians with clustering variables and cluster assignment data 
gayrate_data=data_clean['W1_N1G'].astype('float64')
# split How people rate Gays/lesbians data into train and test sets
gayrate_train, gayrate_test = train_test_split(gayrate_data, test_size=.3, random_state=123)
gayrate_train1=pd.DataFrame(gayrate_train)
gayrate_train1.reset_index(level=0, inplace=True)
merged_train_all=pd.merge(gayrate_train1, merged_train, on='index')
sub1 = merged_train_all[['W1_N1G', 'cluster']].dropna()


import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

gpamod = smf.ols(formula='W1_N1G ~ C(cluster)', data=sub1).fit()
print (gpamod.summary())

print ('means for Gay Rate by cluster')
m1= sub1.groupby('cluster').mean()
print (m1)

print ('standard deviations for Gay Rate by cluster')
m2= sub1.groupby('cluster').std()
print (m2)

mc1 = multi.MultiComparison(sub1['W1_N1G'], sub1['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())









"""
VALIDATION
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster in test data set
"""
# create a variable out of the index for the cluster training dataframe to merge on
clus_test.reset_index(level = 1, inplace = True)
# create a list that has the new index variable
cluslistval=list(clus_test['index'])
# create a list of cluster assignments
labelsval=list(clusassignval)
# combine index variable list with labels list into a dictionary
newlistval=dict(zip(cluslistval, clusassignval))
newlistval
# convert newlist dictionary to a dataframe
newclusval=DataFrame.from_dict(newlistval, orient='index')
newclusval
# rename the cluster assignment column
newclusval.columns = ['cluster']
# create a variable out of the index for the cluster assignment dataframe to merge on
newclusval.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_test=pd.merge(clus_test, newclusval, on='index')
# cluster frequencies
merged_test.cluster.value_counts()
"""
END multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""

# calculate test data clustering variable means by cluster
clustergrpval = merged_test.groupby('cluster').mean()
print ("Test data clustering variable means by cluster")
print(clustergrpval)
