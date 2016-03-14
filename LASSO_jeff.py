# -*- coding: utf-8 -*-
"""
Created on 13 March 2016

@author: Jeff Hsueh
"""

#from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV
 
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

data_clean['have_sex_outside_race']= data_clean['W1_E7'].map(recode1)
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
        
data_clean['same_sex_agree'] = data_clean.apply(lambda row : same_sex_agree(row),axis=1)
#print (data_clean['same_sex_agree'])

data_clean['god_is_anger'] = data_clean.apply(lambda row : god_is_anger(row),axis=1)
#print (data_clean['god_is_anger'])

data_clean['is_highschool'] = data_clean.apply(lambda row : is_highschool(row),axis=1)
#print (data_clean['is_highschool'])

data_clean['is_male'] = data_clean.apply(lambda row : is_male(row),axis=1)
#print (data_clean['is_male'])

data_clean['is_white'] = data_clean.apply(lambda row : is_white(row),axis=1)
#print (data_clean['is_white'])

data_clean['is_black'] = data_clean.apply(lambda row : is_black(row),axis=1)
#print (data_clean['is_black'])

data_clean= data_clean.dropna()
data_clean.dtypes
data_clean.describe()
#Split into training and testing sets
predictors = data_clean[['W1_L2_3', 'W1_L2_4', 'W1_L2_5','W1_F4_A', 'god_is_anger', 
'is_highschool', 'is_male' , 'PPAGECT4', 'PPNET', 'PPINCIMP','W1_P2', 'is_black', 'is_white',
'is_repubilcan', 'is_democrat', 'is_inde','have_sex_outside_race', 
'is_arrested', 'is_unemployed','have_child']]

#'W2_QL2C', 'W2_QL3', 'W2_QL4','W2_QL2A'
# willing_to_date_out'
#

target = data_clean.same_sex_agree

# standardize predictors to have mean=0 and sd=1
#predictors=predvar.copy()
from sklearn import preprocessing
predictors['W1_L2_3']=preprocessing.scale(predictors['W1_L2_3'].astype('float64'))
predictors['W1_L2_4']=preprocessing.scale(predictors['W1_L2_4'].astype('float64'))
predictors['god_is_anger']=preprocessing.scale(predictors['god_is_anger'].astype('float64'))
predictors['is_highschool']=preprocessing.scale(predictors['is_highschool'].astype('float64'))
predictors['is_male']=preprocessing.scale(predictors['is_male'].astype('float64'))
predictors['PPAGECT4']=preprocessing.scale(predictors['PPAGECT4'].astype('float64'))
predictors['PPNET']=preprocessing.scale(predictors['PPNET'].astype('float64'))
predictors['PPINCIMP']=preprocessing.scale(predictors['PPINCIMP'].astype('float64'))
predictors['is_black']=preprocessing.scale(predictors['is_black'].astype('float64'))
predictors['is_white']=preprocessing.scale(predictors['is_white'].astype('float64'))
predictors['W1_P2']=preprocessing.scale(predictors['W1_P2'].astype('float64'))
predictors['W1_F4_A']=preprocessing.scale(predictors['W1_F4_A'].astype('float64'))
predictors['is_repubilcan']=preprocessing.scale(predictors['is_repubilcan'].astype('float64'))
predictors['is_democrat']=preprocessing.scale(predictors['is_democrat'].astype('float64'))
predictors['is_inde']=preprocessing.scale(predictors['is_inde'].astype('float64'))
predictors['have_sex_outside_race']=preprocessing.scale(predictors['have_sex_outside_race'].astype('float64'))
predictors['have_child']=preprocessing.scale(predictors['have_child'].astype('float64'))
predictors['is_arrested']=preprocessing.scale(predictors['is_arrested'].astype('float64'))
predictors['is_unemployed']=preprocessing.scale(predictors['is_unemployed'].astype('float64'))
#predictors['willing_to_date_out']=preprocessing.scale(predictors['willing_to_date_out'].astype('float64'))
#predictors['W2_QL2A']=preprocessing.scale(predictors['W2_QL2A'].astype('float64'))
#predictors['W2_QL2C']=preprocessing.scale(predictors['W2_QL2C'].astype('float64'))
#predictors['W2_QL3']=preprocessing.scale(predictors['W2_QL3'].astype('float64'))
#predictors['W2_QL4']=preprocessing.scale(predictors['W2_QL4'].astype('float64'))
#







""" Binary Variables
#W1_C1: Generally speaking, do you usually think of yourself as a Democrat, a Republican, an
Independent, or something else?
#W1_J2: Which of the following statements comes closest to your view concerning same-sex couples?
#W1_L2_2: [National Urban League]
#W1_L2_3: [Southern Christian Leadership Conference]
#W1_L2_4: [Tea Party Movement]
#W1_L2_5: [Occupy Wall Street Movement]
#W1_M13_B: [God is angered by human sin. ]
#PPEDUCAT: Education (Categorical)
#PPGENDER: Gender
W2_QL4: Do you favor or oppose laws to protect homosexuals against job discrimination?
#PPAGECT4: Age - 4 Categories
#PPNET: HH Internet Access
#W1_E4: Were you ever willing to date outside of your racial group?
#W1_E2: How optimistic are you that you will develop a serious and/or marital relationship? --Target
#W1_E7: Have you had sex with someone outside of your racial group?
#W1_P9: Has anyone in your household ever been arrested for a crime?
#W1_P11: Is anyone in your household currently unemployed?
#W1_P17: Do you have any biological or adopted children?
"""

""" quantitative Variables
#W2_QL2A: [Work with children ] Gays and lesbians should be allowed to
#W2_QL2C: [Have and Raise Children ] Gays and lesbians should be allowed to
#W2_QL3: Some people believe that homosexuality is immoral. To what extent do you agree or
disagree with this perspective?
#W2_QL4: Do you favor or oppose laws to protect homosexuals against job discrimination?
#W1_P2: People talk about social classes such as the poor, the working class, the middle class,
the upper-middle class, and the upper class. Which of these classes would you say you
belong to?
#PPINCIMP: Household Income
#W1_F4_A: [To own a home ] For yourself and people like you, how easy or hard is it to reach
these goals?
"""




# split data into train and test sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, 
                                                              test_size=.3, random_state=123)

# specify the lasso regression model
model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)

# print variable names and regression coefficients
print (dict(zip(predictors.columns, model.coef_)))

# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
         

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)
