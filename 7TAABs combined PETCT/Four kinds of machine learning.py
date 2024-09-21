import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.utils import resample
from lifelines.utils import concordance_index
from scipy.stats import ttest_ind, ranksums
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

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
# model1
alphas_arr = np.logspace(-3, 2, 200, base = 10)
selector_lasso = LassoCV(alphas = alphas_arr
                         , cv = 5 
                         , max_iter = 1e6
                        , n_jobs = -1 
                        )
model1 = selector_lasso
# model2
clf_lr = LogisticRegression(penalty = "l1"
                            , solver="liblinear" 
                            , max_iter=1000
                            , n_jobs = -1
                            #, random_state = 11 # test
                            , multi_class = 'auto'
                           )
model2 = clf_lr
# model3
svc = SVC(kernel='rbf'
          , probability=True
          , C = 2.3330580791522335
          , gamma = 0.004641588833612777
          )
model3 = svc
# model4
clf_xgb = XGBClassifier(max_depth=3,
                    learning_rate=0.1,
                    n_estimators=10,
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
model4 = clf_xgb

model1.fit(X, y)
model2.fit(X, y)
model3.fit(X, y)
model4.fit(X, y)

# Calculation model result
def evaluate_model(y_pred_score,y, i = ''):
    # The AUC is calculated using the bootstraps method
    n_bootstraps = 1000
    auc_scores = []
    for _ in range(n_bootstraps):
        X_boot, y_boot = resample(y_pred_score, y)
        fpr, tpr, _ = roc_curve(y_boot, X_boot)
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)    
    # Calculation 95CI
    auc_scores_sorted = np.sort(auc_scores)
    ci_lower = auc_scores_sorted[int(0.025 * n_bootstraps)]
    ci_upper = auc_scores_sorted[int(0.975 * n_bootstraps)]
    # Calculate the average AUC value
    roc_auc = np.mean(auc_scores)
    print("Model %s：Mean AUC: %.4f, 95%% Confidence Interval: [%.4f, %.4f]" % (i, roc_auc, ci_lower, ci_upper))
    fpr, tpr, thresholds = roc_curve(y, y_pred_score)
    plt.plot(fpr, tpr, lw=lw, label='%s ROC curve (area = %0.2f)' % (i, roc_auc))
    
    # Calculate model performance
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = np.where(y_pred_score >= optimal_threshold, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    ppv = tp / (tp + fp)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred) 
    youden_j = sensitivity + specificity - 1
    c_index = concordance_index(y, y_pred_score)
    
    # Save result
    model_results = pd.DataFrame({
        "Name": ["%s" % (i)],
        "Optimal Threshold": [optimal_threshold],
        "AUC": [roc_auc],
        "95% CI Lower": [ci_lower],
        "95% CI Upper": [ci_upper],
        "Accuracy": [accuracy],
        "Sensitivity": [sensitivity],
        "Specificity": [specificity],
        "PPV": [ppv],
        "NPV": [npv],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
        "Youden's J": [youden_j],
        "c_index" : [c_index]
    })
    
    return model_results, optimal_threshold
    

# Training group
y_pred_score1 = model1.predict(X) # lasso
y_pred_score2 = model2.predict_proba(X)[:,1]
y_pred_score3 = model3.predict_proba(X)[:,1]
y_pred_score4 = model4.predict_proba(X)[:,1]

plt.figure()
lw = 2

model1_results, optimal_threshold1 = evaluate_model(y_pred_score1,y, i = 'Lasso')
model2_results, optimal_threshold2 = evaluate_model(y_pred_score2,y, i = 'LogiReg')
model3_results, optimal_threshold3 = evaluate_model(y_pred_score3,y, i = 'SVM')
model4_results, optimal_threshold4 = evaluate_model(y_pred_score4,y, i = 'XGB')

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

results = pd.DataFrame()
results = results.append(model1_results, ignore_index=True)
results = results.append(model2_results, ignore_index=True)
results = results.append(model3_results, ignore_index=True)
results = results.append(model4_results, ignore_index=True)

# Testing group
test_y_pred_score1 = model1.predict(X_test)
test_y_pred_score2 = model2.predict_proba(X_test)[:,1]
test_y_pred_score3 = model3.predict_proba(X_test)[:,1]
test_y_pred_score4 = model4.predict_proba(X_test)[:,1]

plt.figure()
lw = 2

test_model1_results, test_optimal_threshold1 = evaluate_model(test_y_pred_score1,y_test, i = 'Lasso')
test_model2_results, test_optimal_threshold2 = evaluate_model(test_y_pred_score2,y_test, i = 'LogiReg')
test_model3_results, test_optimal_threshold3 = evaluate_model(test_y_pred_score3,y_test, i = 'SVM')
test_model4_results, test_optimal_threshold4 = evaluate_model(test_y_pred_score4,y_test, i = 'XGB')

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Test)')
plt.legend(loc="lower right")
plt.show()

test_results = pd.DataFrame()
test_results = test_results.append(test_model1_results, ignore_index=True)
test_results = test_results.append(test_model2_results, ignore_index=True)
test_results = test_results.append(test_model3_results, ignore_index=True)
test_results = test_results.append(test_model4_results, ignore_index=True)

total_results = pd.DataFrame()
total_results = total_results.append(results, ignore_index=True)
total_results = total_results.append(test_results, ignore_index=True)
print(total_results)
