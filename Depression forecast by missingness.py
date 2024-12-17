import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import (f_classif,SelectKBest)
import scipy.stats as stats
from scipy.cluster import hierarchy
import warnings
import os
import sys
sys.path.insert(0, '/libraries')
from libraries import Transformation_library
os.environ["PYTHONHASHSEED"] = "0"
###parameters
test = 0.2 #fraction for testing
test_sets = int(1/test) #number of testing sets (5)
val = 0.2 #fractiion for validation (from the rest part)
val_sets = int(1/val) #number of validation sets (5)
level = 0.5 #auc when guess is random
binary = 11 #threshold for depression
ITER = 5 #number of iteration for random forest
n_features = 2 #the maximum number of features we consider
is_balanced = 0 #do we want to use weights during classification?
is_median = 1 #do we want to use median or mean when select optimal n (number of features)
if is_balanced == 1:
    class_weight = "balanced"
else:
    class_weight = None
###metadata
Metadata=pd.read_csv("BASIC_metadata_full.csv",sep=',',low_memory=False) #read file with depression levels and ids of participants
Metadata.loc[Metadata.TimePoint=="Trimester2","TimePoint"] = 0 #timepoits are 0,1,2
Metadata.loc[Metadata.TimePoint=="Trimester3","TimePoint"] = 1
Metadata.loc[Metadata.TimePoint=="PostpartumWeek6","TimePoint"] = 2
i = Metadata[Metadata.ReadsNumber<500000].index #remove insufficient reads
Metadata = Metadata.replace(Metadata.loc[i,'ReadsNumber'],np.nan)
EPDS = 2*np.ones_like(Metadata.EPDS) #2 is for missingness
EPDS[Metadata.EPDS>binary] = 1 #binary labels for depression
EPDS[Metadata.EPDS<=binary] = 0
###data
X_full, labels = [[] for _ in range(2)] #X is 2 values denoting missingness, labels are EPDS at tp 2
Individual_ID = Metadata.Individual_ID #id of a person
for i in np.unique(Individual_ID):
    TP = Metadata.TimePoint[Individual_ID == i].to_numpy()
    if sum((TP - np.array([0,1,2]))**2)!=0:
        warnings.warn("TP are missing or in incorrect order") #does not happen in the dataset we have
    Outcomes = EPDS[Individual_ID == i]
    if Outcomes[2]!=2:
        Reads = Metadata.ReadsNumber[Individual_ID == i].to_numpy()
        labels = np.append(labels, Outcomes[2])
        Dataset_line = np.array([1-np.isnan(Reads[0]), 1-np.isnan(Reads[1])]).reshape(1,2)
        X_full = Dataset_line if np.size(X_full) == 0 else np.concatenate([X_full, Dataset_line])
X_f, X_t = X_full[labels == 0], X_full[labels == 1] #f = healthy, t = depressed
size_f, size_t = np.shape(X_f)[0], np.shape(X_t)[0]
print("sizes of complete f and t sets:",size_f,size_t)
test_f, test_t = int(size_f*test), int(size_t*test) #amounts to put to test set
print("sizes of test sets",test_f, test_t)
AUC_TEST, N_OPT, SENSITIVITY, SPECIFICITY = [[] for _ in range(4)]
aucs_val = np.zeros((val_sets, ITER, n_features))
for TS in range(test_sets): #test sets
    print("test group", TS)
    test_f_range, test_t_range = range(TS * test_f, (TS + 1) * test_f), range(TS * test_t, (TS + 1) * test_t)
    rest_f_range, rest_t_range = np.setdiff1d(range(size_f), test_f_range), np.setdiff1d(range(size_t), test_t_range)
    X_f_test, X_t_test = X_f[test_f_range], X_t[test_t_range]
    X_test = np.concatenate([X_f_test,X_t_test])
    label_test = np.zeros(test_f + test_t)
    label_test[test_f:] = 1
    rest_f, rest_t = np.size(rest_f_range), np.size(rest_t_range)
    val_f, val_t = int(rest_f * val), int(rest_t * val) #amounts to put to validation set
    print("sizes of val sets",val_f, val_t)
    train_f, train_t = rest_f - val_f, rest_t - val_t #the rest of rest is for training set
    print("sizes of train sets", train_f, train_t)
    for VS in range(val_sets): #val sets
        print("val group", VS)
        val_f_range, val_t_range = rest_f_range[range(VS* val_f, (VS + 1) * val_f)], rest_t_range[range(VS* val_t, (VS + 1) * val_t)]
        train_f_range, train_t_range = np.setdiff1d(rest_f_range, val_f_range), np.setdiff1d(rest_t_range, val_t_range)
        X_f_val, X_t_val, X_f_train, X_t_train = X_f[val_f_range], X_t[val_t_range], X_f[train_f_range], X_t[train_t_range]
        X_val, X_train = np.concatenate([X_f_val, X_t_val]), np.concatenate([X_f_train, X_t_train])
        label_val, label_train = np.zeros(val_f + val_t), np.zeros(train_f + train_t)
        label_val[val_f:], label_train[train_f:] = 1, 1
        stats_train, stats_val = X_train, X_val
        for j in range(n_features): #choose the best j + 1 features using filter method
            sel = SelectKBest(f_classif, k=j + 1).fit(pd.DataFrame(stats_train), label_train)
            Index = sel.get_support()
            for seed in range(ITER): #classify by random forest with different random seeds
                rf = RandomForestClassifier(random_state = seed, class_weight=class_weight).fit(stats_train[:,Index], label_train)
                y_pred = rf.predict(stats_val[:,Index])
                aucs_val[VS, seed, j] = roc_auc_score(label_val, y_pred)
    if is_median == 1: #optimal number of features by median of mean
        print("medians", np.median(aucs_val, axis=(0, 1))) #average by validation sets and random seeds
        n_opt = np.argmax(np.median(aucs_val, axis=(0, 1))) + 1
    else:
        print("means", np.mean(aucs_val, axis = (0,1)))
        n_opt=np.argmax(np.mean(aucs_val, axis = (0,1))) + 1
    N_OPT =np.append(N_OPT, n_opt)
    X_rest, label_rest = np.concatenate([X_train, X_val]), np.concatenate([label_train,label_val])#concatenate everything except testing set
    stats_rest, stats_test = X_rest, X_test
    sel = SelectKBest(f_classif, k=n_opt).fit(pd.DataFrame(stats_rest), label_rest) #select optimal number of features
    Index = sel.get_support()
    print("test features:",sel.get_feature_names_out()) #from features namas we can reconstruct what bacteria we use
    for seed in range(ITER):
        rf = RandomForestClassifier(random_state = seed, class_weight=class_weight).fit(stats_rest[:, Index], label_rest)
        print("importances",rf.feature_importances_) #display importances of features
        y_pred = rf.predict(stats_test[:, Index])
        print("auc", roc_auc_score(label_test, y_pred)) #display balanced accuracy
        AUC_TEST = np.append(AUC_TEST, roc_auc_score(label_test, y_pred)) # balanced accuracy on testing set
        SENSITIVITY = np.append(SENSITIVITY, sum(y_pred[label_test==1]==1)/sum(label_test==1))
        SPECIFICITY = np.append(SPECIFICITY, sum(y_pred[label_test==0]==0)/sum(label_test==0))
print("mean auc",np.mean(AUC_TEST))
result= stats.ttest_1samp(AUC_TEST, level, alternative="greater")
print("p value",result.pvalue)
print("true positive",np.mean(SENSITIVITY))
print("true negative",np.mean(SPECIFICITY))
print("n opt",np.mean(N_OPT))
print(N_OPT)