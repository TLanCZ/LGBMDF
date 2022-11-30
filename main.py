# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 21:22:46 2022

@author: Administrator
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import average_precision_score,confusion_matrix
import scipy.io as sio
from gcforest.gcforest import GCForest
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import xgboost
import pandas as pd
# import catboost  
import gcforest
import time
import random
import lightgbm



#2LGB 2RF
# def get_toy_config():
#     config = {}
#     ca_config = {}
#     ca_config["random_state"] = 0
#     ca_config["max_layers"] = 100
#     ca_config["early_stopping_rounds"] = 3
#     ca_config["n_classes"] = 2
#     ca_config["estimators"] = []
    
#     ca_config["estimators"].append({"n_folds": 5, "type": "LGBMClassifier",'objective':'binary',     
#           "num_leaves":  200, "n_estimators": 400,"max_depth": 11, 'learning_rate':0.1, "n_jobs": -1 })
#     ca_config["estimators"].append({"n_folds": 5, "type": "LGBMClassifier",'objective':'binary',     
#           "num_leaves":  200, "n_estimators": 400,"max_depth": 11, 'learning_rate':0.1, "n_jobs": -1 })
#     ca_config["estimators"].append({"n_folds": 5, "type": "LGBMClassifier",'objective':'binary',     
#           "num_leaves":  200, "n_estimators": 400,"max_depth": 11, 'learning_rate':0.1, "n_jobs": -1 })
    
    
#     ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 400, "max_depth": None, "n_jobs": -1})
#     ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 400, "max_depth": None, "n_jobs": -1})
#     ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 400, "max_depth": None, "n_jobs": -1})
    
#     config["cascade"] = ca_config
#     return config

#2lgbm 2rf 2et
# def get_toy_config():
#     config = {}
#     ca_config = {}
#     ca_config["random_state"] = 0
#     ca_config["max_layers"] = 100
#     ca_config["early_stopping_rounds"] = 3
#     ca_config["n_classes"] = 2
#     ca_config["estimators"] = []
    
#     ca_config["estimators"].append({"n_folds": 5, "type": "LGBMClassifier",'objective':'binary',     
#           "num_leaves":  200, "n_estimators": 400,"max_depth": 11, 'learning_rate':0.1, "n_jobs": -1 })
#     ca_config["estimators"].append({"n_folds": 5, "type": "LGBMClassifier",'objective':'binary',     
#           "num_leaves":  200, "n_estimators": 400,"max_depth": 11, 'learning_rate':0.1, "n_jobs": -1 })
#     ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500,
#                                     "max_depth": None, "n_jobs": -1})
#     ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, 
#                                     "max_depth": None, "n_jobs": -1})
#     ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 400, 
#                                     "max_depth": None, "n_jobs": -1})
#     ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 400,
#                                     "max_depth": None, "n_jobs": -1})
    
#     config["cascade"] = ca_config
#     return config


#AOPEDF
# def get_toy_config():
#     config = {}
#     ca_config = {}
#     ca_config["random_state"] = 0
#     ca_config["max_layers"] = 100
#     ca_config["early_stopping_rounds"] = 3
#     ca_config["n_classes"] = 2
#     ca_config["estimators"] = []
#     ca_config["estimators"].append(
#             {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 500, "max_depth": 5,
#               "objective": "multi:softprob",  "nthread": -1, "learning_rate": 0.1,"num_class": 2} )
#     ca_config["estimators"].append(
#             {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 500, "max_depth": 5,
#               "objective": "multi:softprob","nthread": -1, "learning_rate": 0.1,"num_class": 2} )
#     ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
#     ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
#     ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
#     ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
#     config["cascade"] = ca_config
#     return config



#3lgb3et
def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 5, "type": "LGBMClassifier",'objective':'binary',     
          "num_leaves":  200, "n_estimators": 400,"max_depth": 11, 'learning_rate':0.1, "n_jobs": -1 })
    ca_config["estimators"].append({"n_folds": 5, "type": "LGBMClassifier",'objective':'binary',     
          "num_leaves":  200, "n_estimators": 400,"max_depth": 11, 'learning_rate':0.1, "n_jobs": -1 })
    ca_config["estimators"].append({"n_folds": 5, "type": "LGBMClassifier",'objective':'binary',     
          "num_leaves":  200, "n_estimators": 400,"max_depth": 11, 'learning_rate':0.1, "n_jobs": -1 })
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    config["cascade"] = ca_config
    return config


drugFeature=np.loadtxt('drugFeature.txt')
proteinFeature=np.loadtxt('proteinFeature.txt')
interaction=np.loadtxt('dataset/Networks/drugProtein.txt')







positive_feature=[]
negative_feature=[]
alldata=[]
for i in range(np.shape(interaction)[0]):
    for j in range(np.shape(interaction)[1]):
        temp=np.append(drugFeature[i],proteinFeature[j])
        if int(interaction[i][j])==1:
            positive_feature.append(temp)
        elif int(interaction[i][j])==0:
            negative_feature.append(temp)
            
            
from imblearn.under_sampling import RandomUnderSampler#随机欠采样

x = np.vstack((positive_feature,negative_feature))

labela=np.ones((len(positive_feature),1))
labelb=np.zeros((len(negative_feature),1))
y=np.vstack((labela,labelb))


rus= RandomUnderSampler(random_state = 7)
feature, label = rus.fit_resample(x, y)







# feature = np.load('feature.npy')
# label = np.load('label.npy')




# rs = np.random.randint(0, 1000, 1)[0]
kf = KFold( n_splits=5, shuffle=True, random_state = 119)


start =time.time()

test_SN_fold = []
test_SP_fold = []
test_MCC_fold = []

test_precision_fold = []
test_auc_fold = []
test_aupr_fold = []
for train_index, test_index in kf.split(label[:,0]):
    Xtrain, Xtest = feature[train_index], feature[test_index]
    Ytrain, Ytest = label[train_index], label[test_index]

    config = get_toy_config()
    rf=GCForest(config)
    Ytrain=Ytrain.flatten()
    rf.fit_transform(Xtrain, Ytrain)
    
    
    # deep forest
    predict_y = rf.predict(Xtest)
    acc = accuracy_score(Ytest, predict_y)
    print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))
    prob_predict_y = rf.predict_proba(Xtest)  # Give a result with probability values，the probability sum is 1 
    predictions_validation = prob_predict_y[:, 1]
    TN, FP, FN, TP = confusion_matrix(Ytest,predict_y).ravel()
    precision = TP/(TP+FP)
   
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    fz = TP * TN - FP * FN
    fm = (TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)
    MCC = fz / pow(fm, 0.5)
    fpr, tpr, _ = roc_curve(Ytest, predictions_validation)
    roc_auc = auc(fpr, tpr)
    aupr = average_precision_score(Ytest, predictions_validation)
    
    print(roc_auc)
    print(aupr)
    test_SN_fold.append(SN)
    test_SP_fold.append(SP)
    test_MCC_fold.append(MCC)
    test_auc_fold.append(roc_auc)
    test_aupr_fold.append(aupr)
    
    test_precision_fold.append(precision)
    
    
mean_SN = np.mean(test_SN_fold)
mean_SP = np.mean(test_SP_fold)
mean_MCC = np.mean(test_MCC_fold)
mean_auc=np.mean(test_auc_fold)
mean_pr=np.mean(test_aupr_fold)


print('mean auc aupr', mean_auc, mean_pr)
print('mean sn sp mcc',mean_SN,mean_SP,mean_MCC)

end=time.time()
print('Running time: %s Seconds'%(end-start))



