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
Cs = np.logspace(-1,3,10,base = 2)
gammas = np.logspace(-5,1,10)
param_grid = {'C': Cs, 'gamma': gammas, 'kernel': ('rbf','linear')}

grid_search = GridSearchCV(SVC(probability=True)
                  , param_grid = param_grid
                  , cv = 5
                  , n_jobs = -1
                  )
grid_search.fit(X,y)

print('best：')
print(grid_search.best_params_)
