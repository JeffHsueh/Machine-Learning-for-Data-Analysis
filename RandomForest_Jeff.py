# Editor : Jeff Hsueh
# Last Edit Date: 06-Mar. 2016

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
 # Feature Importance
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier

#os.chdir("C:\TREES")

#Load the dataset

AH_data = pd.read_csv("tree_ool_pds.csv")
data_clean = AH_data.dropna()
#data_clean.dtypes
#data_clean.describe()

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

def is_male (row) : # Agree = 1 ; Disagree = 0
    if row ['PPGENDER'] == 1 :
        return'1'
    elif row ['PPGENDER'] == 2 :
        return "0"
    else:
        return np.nan
        
def is_white(row):
    if row['PPETHM']  == 1 :
        return "1"
    else :
        return "0"
def is_black(row):
    if row["PPETHM"] == 2 :
        return "1"
    else: 
        return "0"
        
data_clean['same_sex_agree'] = data_clean.apply(lambda row : same_sex_agree(row),axis=1)
print (data_clean['same_sex_agree'])

data_clean['god_is_anger'] = data_clean.apply(lambda row : god_is_anger(row),axis=1)
print (data_clean['god_is_anger'])

data_clean['is_highschool'] = data_clean.apply(lambda row : is_highschool(row),axis=1)
print (data_clean['is_highschool'])

data_clean['is_male'] = data_clean.apply(lambda row : is_male(row),axis=1)
print (data_clean['is_male'])

data_clean['is_white'] = data_clean.apply(lambda row : is_white(row),axis=1)
print (data_clean['is_white'])

data_clean['is_black'] = data_clean.apply(lambda row : is_black(row),axis=1)
print (data_clean['is_black'])

data_clean = data_clean.dropna()
data_clean.dtypes
data_clean.describe()


#W1_J2: Which of the following statements comes closest to your view concerning same-sex couples?

#W1_L2_2: [National Urban League]
#W1_L2_3: [Southern Christian Leadership Conference]
#W1_L2_4: [Tea Party Movement]
#W1_L2_5: [Occupy Wall Street Movement]
#W1_M13_B: [God is angered by human sin. ]
#PPEDUCAT: Education (Categorical)
#PPGENDER: Gender

#Split into training and testing sets
predictors = data_clean[['W1_L2_3', 'W1_L2_4', 'W1_L2_5', 'god_is_anger', 'same_sex_agree', 
'is_highschool', 'is_male']]
targets = data_clean.W1_J2

pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, targets, test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
print(sklearn.metrics.accuracy_score(tar_test, predictions))


# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)
# display the relative importance of each attribute
print(model.feature_importances_)


"""
Running a different number of trees and see the effect
 of that on the accuracy of the prediction
"""

trees=range(25)
accuracy=np.zeros(25)


for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)
   #print (accuracy[idx])

 
plt.cla()
plt.plot(trees, accuracy)

