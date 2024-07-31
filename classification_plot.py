import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import (f_classif,SelectKBest)
import random
import sys
sys.path.insert(0, '/libraries')
from libraries import Transformation_library, Imputation_library
import os
os.environ["PYTHONHASHSEED"] = "0"
###parameters
is_sparse = 1 #do we want to reduce sparsity?
is_balanced = 0 #do we want to use weights during classification
is_median = 0 #do we want to use median or mean when select optimal n (number of features)
val_sets = 5 #number of validation sets
level = 0.5 #auc when guess is random
binary = 11 #threshold for depression
n_features = 20 #the maximum number of microbiome features, can be 3*d at max
sparsity_level = 0.5 #sparsity thershold to merge bacteria together
if is_balanced == 1:
    class_weight = "balanced"
else:
    class_weight = None
###metadata
Metadata=pd.read_csv("BASIC_metadata.csv",sep=',',low_memory=False)
Metadata.loc[Metadata.TimePoint=="Trimester2","TimePoint"] = 0 #timepoits are 0,1,2
Metadata.loc[Metadata.TimePoint=="Trimester3","TimePoint"] = 1
Metadata.loc[Metadata.TimePoint=="PostpartumWeek6","TimePoint"] = 2
i = Metadata[pd.Series.isna(Metadata.Sample_ID)].index #remove NaN values
Metadata = Metadata.drop(i)
i = Metadata[Metadata.ReadsNumber<500000].index #remove insufficient reads
Metadata = Metadata.drop(i)
EPDS = np.zeros_like(Metadata.EPDS)
EPDS[Metadata.EPDS>binary] = 1 #binary labels for depression
### species
profile =pd.read_csv("Species_Profile.csv",sep=',',low_memory=False)
ID = profile.Sample_id.to_numpy() #id of a sample in profile
species=profile.to_numpy()[:,1:]/100 #normilized to 1
d_full = np.shape(species)[1]
X_full, X_02, X_12, X_2, labels, labels_02, labels_12, labels_2 = [[] for _ in range(8)] #X is data in 3 rows, labels are EPDS at tp 2
Individual_ID = Metadata.Individual_ID #id of a person
for i in np.unique(Individual_ID):
    TP = Metadata.TimePoint[Individual_ID == i].to_numpy()
    Depression = EPDS[Individual_ID == i]
    Sample_ID = Metadata.Sample_ID[Individual_ID == i] #id of a sample in metadata
    if np.sum(TP == 2) == 1: #for cases of missingness, but tp 2 is required since it contains the label
        if np.sum(TP == 0)==1 and np.sum(TP == 1)==1: #complete data
            labels = np.append(labels, Depression[TP==2][0])
            Dataset_line = np.zeros((1,3,d_full))
            for j in [0,1,2]:
                id = Sample_ID[TP == j].to_numpy()
                Dataset_line[:, j, :] = species[np.where(ID == id)[0], :]
            X_full = Dataset_line if np.size(X_full) == 0 else np.concatenate([X_full, Dataset_line])
        if np.sum(TP == 0)==0 and np.sum(TP == 1)==1: #tp 0 is missing
            labels_12 = np.append(labels_12, Depression[TP==2][0])
            Dataset_line = np.zeros((1,3,d_full))
            for j in [1,2]:
                id = Sample_ID[TP == j].to_numpy()
                Dataset_line[:, j, :] = species[np.where(ID == id)[0], :]
            X_12 = Dataset_line if np.size(X_12) == 0 else np.concatenate([X_12, Dataset_line])
        if np.sum(TP == 0)==1 and np.sum(TP == 1)==0: #tp 1 is missing
            labels_02 = np.append(labels_02, Depression[TP==2][0])
            Dataset_line = np.zeros((1,3,d_full))
            for j in [0,2]:
                id = Sample_ID[TP == j].to_numpy()
                Dataset_line[:, j, :] = species[np.where(ID == id)[0], :]
            X_02 = Dataset_line if np.size(X_02) == 0 else np.concatenate([X_02, Dataset_line])
        else:
            labels_2 = np.append(labels_2, Depression[TP==2][0])
            Dataset_line = np.zeros((1,3,d_full))
            id = Sample_ID[TP == 2].to_numpy()
            Dataset_line[:, 2, :] = species[np.where(ID == id)[0], :]
            X_2 = Dataset_line if np.size(X_2) == 0 else np.concatenate([X_2, Dataset_line])
X_f, X_t = X_full[labels == 0], X_full[labels == 1] #f = healthy, t = true
size_f, size_t = np.shape(X_f)[0], np.shape(X_t)[0]
print("sizes of complete f and t sets:",size_f,size_t)
AUC_TEST_mean = []
TRUE_POSITIVE_mean = []
TRUE_NEGATIVE_mean = []
AUC_TEST_std = []
TRUE_POSITIVE_std = []
TRUE_NEGATIVE_std = []
for n_in_sets in range(1,8):
    AUC_TEST = []
    TRUE_POSITIVE = []
    TRUE_NEGATIVE = []
    for iter in range(125):
        test_f_range, test_t_range = random.sample(range(size_f), n_in_sets), random.sample(range(size_t), n_in_sets)
        rest_f_range, rest_t_range = np.setdiff1d(range(size_f), test_f_range), np.setdiff1d(range(size_t), test_t_range)
        X_f_test, X_t_test = X_f[test_f_range], X_t[test_t_range]
        X_test = np.concatenate([X_f_test,X_t_test])
        label_test = np.zeros(2*n_in_sets)
        label_test[n_in_sets:] = 1
        rest_f, rest_t = np.size(rest_f_range), np.size(rest_t_range)
        train_f, train_t = rest_f - n_in_sets, rest_t - n_in_sets #the rest of rest is for training set
        aucs_val = np.zeros((val_sets, n_features))
        for VS in range(val_sets): #val sets
            print("val group", VS)
            val_f_range, val_t_range = rest_f_range[random.sample(range(rest_f), n_in_sets)], rest_t_range[random.sample(range(rest_t), n_in_sets)]
            train_f_range, train_t_range = np.setdiff1d(rest_f_range, val_f_range), np.setdiff1d(rest_t_range, val_t_range)
            X_f_val, X_t_val, X_f_train, X_t_train = X_f[val_f_range], X_t[val_t_range], X_f[train_f_range], X_t[train_t_range]
            X_val, X_train = np.concatenate([X_f_val, X_t_val]), np.concatenate([X_f_train, X_t_train])
            label_val, label_train = np.zeros(2*n_in_sets), np.zeros(train_f + train_t)
            label_val[n_in_sets:], label_train[train_f:] = 1, 1
            X_test_cut, X_12_cut, X_02_cut, X_2_cut = X_test, X_12, X_02, X_2 #no changes in data dimensionality if is_sparse = False
            X = [X_train, X_val, X_test_cut, X_12_cut, X_02_cut, X_2_cut]
            if is_sparse == 1:
                X_combine = np.concatenate([X_train.reshape(-1,d_full), X_02[:,[0,2],:].reshape(-1,d_full), X_12[:,[1,2],:].reshape(-1,d_full), X_2[:,[2],:].reshape(-1,d_full)])
                sparsity = np.sum(X_combine == 0, axis=0) / np.shape(X_combine)[0]
                for xi in range(len(X)):
                    other_species = np.sum(X[xi][:, :, sparsity >= sparsity_level], axis=-1).reshape(-1, 3, 1)
                    X[xi] = X[xi][:, :, sparsity < sparsity_level]
                    X[xi]= np.concatenate([X[xi], other_species], axis=-1)
            d = np.shape(X[0])[2]
            print("d=", d)
            dY = 3 * d
            delta = np.min(X[0][X[0] > 0])
            print("delta", delta)  # the smallest element
            for xi in range(len(X)):
                X[xi] = (X[xi] + delta) / (1 + delta * d) #remove 0s
            X_train, X_val, X_test_cut, X_12_cut, X_02_cut, X_2_cut = X
            ###classes of missingness
            class_train, class_val, class_test = np.ones((np.shape(X_train)[0],2)), np.ones((np.shape(X_val)[0],2)), np.ones((np.shape(X_test_cut)[0],2))
            class_12, class_02, class_2 = np.zeros((np.shape(X_12_cut)[0],2)), np.zeros((np.shape(X_02_cut)[0],2)), np.zeros((np.shape(X_2_cut)[0],2))
            class_12[:,1], class_02[:,0] = 1, 1
            ###imputation
            D_02, D_12 = np.concatenate([X_train, X_02_cut]), np.concatenate([X_train, X_12_cut]) #D02 is training set for imputing tp 0 from tp 2
            X_12_cut = Imputation_library.impute_one(X_train,X_val,X_12_cut, d, [1,2], 0, None)
            X_02_cut = Imputation_library.impute_one(X_train, X_val, X_02_cut, d, [0,2], 1, None)
            X_2_cut = Imputation_library.impute_two(D_02, D_12, X_val, X_2_cut, 2, [0,1], None)
            X_train, label_train, class_train = np.concatenate([X_train,X_12_cut,X_02_cut,X_2_cut]), np.concatenate([label_train,labels_12,labels_02,labels_2]), np.concatenate([class_train,class_12,class_02,class_2]) #concatenate imputed sets to training set
            X_train, X_val = np.reshape(X_train, (-1, dY)) / 3, np.reshape(X_val, (-1, dY)) / 3 #flatten inputs to one compositional vector
            ### construct hierarchy tree
            Y_train = Transformation_library.CLR(X_train)
            Z = hierarchy.linkage(np.transpose(Y_train), method="ward")
            clustersize, clusters = Transformation_library.hierarchy_tree(Z, dY)
            stats_train, stats_val = Transformation_library.data_transformation(X_train, Z, clustersize, clusters, dY), Transformation_library.data_transformation(X_val, Z, clustersize, clusters, dY) ### construct features from hierarchy tree
            stats_train, stats_val = np.concatenate([class_train, stats_train],axis = 1), np.concatenate([class_val, stats_val],axis = 1)
            for j in range(n_features): #choose the best j + 1 features using fliter method
                sel = SelectKBest(f_classif, k=j + 1).fit(pd.DataFrame(stats_train), label_train)
                Index = sel.get_support()
                rf = RandomForestClassifier(class_weight=class_weight).fit(stats_train[:,Index], label_train)
                y_pred = rf.predict(stats_val[:,Index])
                aucs_val[VS, j] = roc_auc_score(label_val, y_pred)
        if is_median == 1: #optimal number of features by median of mean
            print("medians", np.median(aucs_val, axis=0)) #average by validation sets
            n_opt = np.argmax(np.median(aucs_val, axis=0)) + 1
        else:
            print("means", np.mean(aucs_val, axis = 0))
            n_opt=np.argmax(np.mean(aucs_val, axis = 0)) + 1
        X_rest, label_rest, class_rest = np.concatenate([X_train, X_val]), np.concatenate([label_train,label_val]), np.concatenate([class_train,class_val])#concatenate everything except testing set
        X_test_cut = np.reshape(X_test_cut, (-1, dY)) / 3 #flatten testing set
        Y_rest = Transformation_library.CLR(X_rest)
        Z = hierarchy.linkage(np.transpose(Y_rest), method="ward") #make hierarchy tree again for testing
        clustersize, clusters = Transformation_library.hierarchy_tree(Z, dY)
        stats_rest, stats_test = Transformation_library.data_transformation(X_rest, Z, clustersize, clusters, dY), Transformation_library.data_transformation(X_test_cut, Z, clustersize, clusters, dY)
        stats_rest, stats_test = np.concatenate([class_rest, stats_rest], axis=1), np.concatenate([class_test, stats_test],axis=1)
        sel = SelectKBest(f_classif, k=n_opt).fit(pd.DataFrame(stats_rest), label_rest) #select optimal number of features
        Index = sel.get_support()
        print(sel.get_feature_names_out()) #from features namas we can reconstruct what bacteria we use
        rf = RandomForestClassifier(class_weight=class_weight).fit(stats_rest[:, Index], label_rest)
        print("importances",rf.feature_importances_) #display importances of features
        y_pred = rf.predict(stats_test[:, Index])
        print("auc", roc_auc_score(label_test, y_pred)) #display balanced accuracy
        AUC_TEST = np.append(AUC_TEST, roc_auc_score(label_test, y_pred)) # balanced accuracy on testing set
        TRUE_POSITIVE = np.append(TRUE_POSITIVE, sum(y_pred[label_test==1]==1)/sum(label_test==1))
        TRUE_NEGATIVE = np.append(TRUE_NEGATIVE, sum(y_pred[label_test==0]==0)/sum(label_test==0))
    AUC_TEST_mean, AUC_TEST_std = np.append(AUC_TEST_mean,np.mean(AUC_TEST)), np.append(AUC_TEST_std,np.std(AUC_TEST))
    TRUE_POSITIVE_mean, TRUE_POSITIVE_std = np.append(TRUE_POSITIVE_mean,np.mean(TRUE_POSITIVE)), np.append(TRUE_POSITIVE_std,np.std(TRUE_POSITIVE))
    TRUE_NEGATIVE_mean, TRUE_NEGATIVE_std = np.append(TRUE_NEGATIVE_mean,np.mean(TRUE_NEGATIVE)), np.append(TRUE_NEGATIVE_std,np.std(TRUE_NEGATIVE))
    print(AUC_TEST_mean, AUC_TEST_std, TRUE_POSITIVE_mean, TRUE_POSITIVE_std, TRUE_NEGATIVE_mean, TRUE_NEGATIVE_std)
np.save("AUC_TEST_mean",AUC_TEST_mean)
np.save("AUC_TEST_std",AUC_TEST_std)
np.save("TRUE_POSITIVE_mean",TRUE_POSITIVE_mean)
np.save("TRUE_POSITIVE_std",TRUE_POSITIVE_std)
np.save("TRUE_NEGATIVE_mean",TRUE_NEGATIVE_mean)
np.save("TRUE_NEGATIVE_std",TRUE_NEGATIVE_std)