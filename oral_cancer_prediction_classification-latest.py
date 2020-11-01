# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:12:35 2020

@author: Dr Rashmi S
"""


import numpy as np 
import pandas as pd 



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



data = pd.read_csv('F:/Implementation/Oral Cancer Data/Oral Cancer Consolidated-latest-numeric1.csv')
data.head()
data.info()
data=data.drop(columns=["Donor ID","Project Code","Primary Site","SSM",
                    "CNSM","STSM","SGV","METH-A","METH-S","EXP-A",
                    "EXP-S","PEXP","miRNA-S","JCN"])
data.head()
(data['Survival Time (days)'].value_counts())
data_abs=data[data['Survival Time (days)']=='?']
data_abs=data_abs.reset_index()
data_abs=data_abs.drop(columns=['index'])
data_abs.head()

data_pre=data[data['Survival Time (days)']!='?']
data_pre = data_pre.reset_index()
data_pre = data_pre.drop(columns=["index"])
data_pre = data_pre.astype(np.float64)
data_pre.head()
data_pre_tmp = data_pre.drop(columns=['Survival Time (days)'])

Xin = data_pre_tmp.values
y = data_pre['Survival Time (days)'].values

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(Xin, y, test_size=0.2, random_state=4)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

k_val_min = 2
test_MAE_array = []
k_array = []
MAE = 10^12

for k in range(2, 20):
    model = KNeighborsRegressor(n_neighbors=k).fit(train_x, train_y)
    
    y_predict = model.predict(test_x)
    y_true = test_y

    test_MAE = mean_absolute_error(y_true, y_predict)
    if test_MAE < MAE:
        MAE = test_MAE
        k_val_min = k

    test_MAE_array.append(test_MAE)
    k_array.append(k)

plt.plot(k_array, test_MAE_array,'r')
plt.show()

print("Best k parameter is ",k_val_min )

final_model = KNeighborsRegressor(n_neighbors=16).fit(Xin,y)


data_abs_tmp = data_abs.drop(columns=['Survival Time (days)'])
data_abs_tmp = data_abs_tmp.astype(np.float64)
data_abs_tmp.head()
Xdim = data_abs_tmp.values
ydim = final_model.predict(Xdim)
ydim
ydim = np.round(ydim)
ydim = ydim.astype(np.int64)
ydim
data_predict = pd.DataFrame({'Survival Time (days)':ydim})
data_frame_1 = data_abs_tmp.join(data_predict)
data_frame_1 = data_frame_1.astype(np.int64)
data_frame_1
df_join_2 = data_pre['Survival Time (days)']
data_frame_2 = data_pre_tmp.join(df_join_2)
data_frame_2 = data_frame_2.astype(np.int64)
data_frame_2.head()
data_frame = [data_frame_1, data_frame_2]
data_frame = pd.concat(data_frame)
data_frame.head()

data['Tumor Stage at Diagnosis'].value_counts()
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1)

ax.pie(data['Tumor Stage at Diagnosis'].value_counts(),explode=(0.1,0.1,0.1,0.1), autopct='%1.1f%%',shadow=True, labels = ['T1','T2','T3','T4'],colors=['b','g','c','r'])
plt.axis = 'equal'
data_frame.hist(figsize = (30, 30))
plt.show()

data_frame_1 = data_frame

def num_to_class(x):
    if x==10:
        return 'T1'
    elif x==20:
        return 'T2'
    elif x==30:
        return 'T3'
    elif x==40:
        return 'T4'
    

data_frame_1['Tumor Stage at Diagnosis'] = data_frame_1['Tumor Stage at Diagnosis'].apply(lambda x: num_to_class(x))
data_frame_1['Tumor Stage at Diagnosis'].value_counts()
import seaborn as sns
for i in range(4):
    x = data_frame.iloc[:,i]
    for j in range(i+1,4):
        y = data_frame.iloc[:,j]
        hue_parameter = data_frame['Tumor Stage at Diagnosis']
        ax = sns.scatterplot(x=x, y=y, hue=hue_parameter)
        plt.show()
        
X = data_frame_1.drop(columns='Tumor Stage at Diagnosis').values
Y = data_frame_1['Tumor Stage at Diagnosis'].values

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=4,shuffle=True)
from sklearn.svm import SVC
model = SVC()
model.fit(train_x, train_y)

y_true = test_y
y_predict = model.predict(test_x)
from sklearn.metrics import confusion_matrix
confusion_matrix_1 = confusion_matrix(y_true, y_predict)
print(confusion_matrix_1)
sns.heatmap(confusion_matrix_1, annot=True)
from sklearn.metrics import classification_report
print(classification_report(y_true, y_predict))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_true, y_predict))




