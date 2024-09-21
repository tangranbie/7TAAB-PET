from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Data path
base_Path = r'C:\Users\60288\Desktop\data'
File_name = 'model_data.xlsx'
data_File_Path = os.path.join(base_Path, File_name)

df = pd.read_excel(data_File_Path, index_col='ID')
X_total = df.iloc[: ,1:df.shape[1]]
y_total = df['label']

X, X_test, y, y_test = train_test_split(X_total, y_total, test_size = 0.3,
													# random_state = 11, # test
													stratify = y
													)
# -------------------------------------------------------
scoreTrainList, scoreTestList = [], []
maxTreeNum = 20
# scope = range(1,maxTreeNum)
# for i in range(1,maxTreeNum):
scope = np.arange(0.1, 1.1, 0.1)
for i in np.arange(0.1, 1.1, 0.1): # Multiple parameters can be assigned in sequence
    learn_clf_xgb =  XGBClassifier(max_depth=3,
                    learning_rate= i, 
                    n_estimators= 10, 
                    gamma=0, 
                    subsample=0.5, 
                    objective='binary:logistic', 
                    booster='gbtree',
                    min_child_weight=1,
                    max_delta_step=0,                    
                    colsample_bytree=1,
                    reg_alpha=0,
                    reg_lambda=1
                 )
    learn_clf_xgb.fit(X, y)
    score_test = learn_clf_xgb.score(X_test, y_test)
    score_train = learn_clf_xgb.score(X,y)
    scoreTestList.append(score_test)
    scoreTrainList.append(score_train)
plt.plot(scope, scoreTestList, label = 'Test')
plt.plot(scope, scoreTrainList, label = 'Train')
plt.legend()
plt.show()
