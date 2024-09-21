from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
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
number_Cs = 30
Cs = np.logspace(-3,2,number_Cs,base = 10)
param_grid = {'C': Cs, 'penalty': ['l1', 'l2']}
learn_clf_lr = LogisticRegression(penalty = "l1" 
                                , solver="liblinear" 
                                , max_iter=1000 
                                , n_jobs = -1
                                )
                                
grid_search = GridSearchCV(learn_clf_lr, param_grid, cv=5)
grid_search.fit(X, y)
print('best: ', grid_search.best_params_)

results = pd.DataFrame(grid_search.cv_results_)
scores = np.array(results.mean_test_score).reshape(number_Cs,2)
plt.grid()
plt.plot(param_grid['C'], scores[:,0], 'o-', label='L1 Accuracy',c = 'r' )
plt.plot(Cs, scores[:,1], 'o-', label='L2 Accuracy',c = 'b' )
plt.ylabel('Accuracy')
plt.xscale('log')
plt.legend()
plt.show()
