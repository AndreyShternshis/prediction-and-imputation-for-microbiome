###import packages
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import (f_classif,SelectKBest)
from sklearn.preprocessing import normalize
import scipy.stats as stats
from scipy.cluster import hierarchy
import warnings
import sys
sys.path.insert(0, '/libraries')
from libraries import Transformation_library, Imputation_library
import os
os.environ["PYTHONHASHSEED"] = "0"
import argparse
def script(is_balanced, is_median, Dataset_N, Imputation_type, Imputation_by, Transoformation_type):
    ###standard parameters
    level = 0.5 #auc when guess is random
    test = 0.2 #fraction for testing
    val = 0.2 #fraction for validation (from the rest part)
    ITER = 5 #number of iteration for random forest
    n_features = 20 #the maximum number of features we consider
    test_sets = int(1/test) #number of testing sets (5)
    val_sets = int(1/val) #number of validation sets (5)
    if Dataset_N==1:
        binary = 11 #threshold for depression
    if Dataset_N==2:
        is_sparse = 1 #do we want to reduce sparsity?
        sparsity_level = 0.9 #sparsity threshold to merge bacteria together
    if is_balanced == 1:
        class_weight = "balanced"
    else:
        class_weight = None
    if Imputation_type=="No": #number of imputation sets
        Imputation_sets = 1
    else:
        Imputation_sets = 5
    ###metadata
    if Dataset_N==1:
        Metadata=pd.read_csv("BASIC_metadata_full.csv",sep=',',low_memory=False) #read file with depression levels and ids of participants
        Metadata.loc[Metadata.TimePoint=="Trimester2","TimePoint"] = 0 #timepoits are 0,1,2
        Metadata.loc[Metadata.TimePoint=="Trimester3","TimePoint"] = 1
        Metadata.loc[Metadata.TimePoint=="PostpartumWeek6","TimePoint"] = 2
        i = Metadata[Metadata.ReadsNumber<500000].index #remove insufficient reads
        Metadata = Metadata.replace(Metadata.loc[i,'ReadsNumber'],np.nan)
        TARGET = 2*np.ones_like(Metadata.EPDS) #2 is for missingness
        TARGET[Metadata.EPDS>binary] = 1 #binary labels for depression
        TARGET[Metadata.EPDS<=binary] = 0
    if Dataset_N==2:
        Metadata=pd.read_csv("GMAP_metadata_public.csv",sep=",",low_memory=False)
        Metadata = Metadata.rename(columns={"sample": "tp"})
        Metadata.loc[Metadata.tp=="3_twomonth","tp"] = 0
        Metadata.loc[Metadata.tp=="4_fourmonth","tp"] = 1
        Metadata.loc[Metadata.tp=="5_sixmonth","tp"] = 2
        time = [".3.twomonth", ".4.fourmonth", ".5.sixmonth"]
        i = Metadata[pd.Series.isna(Metadata.case_id)].index #remove NaN values
        Metadata = Metadata.drop(i)
        TARGET = np.zeros_like(Metadata.case_id)
        TARGET[Metadata.case_id.to_numpy()=="AP Case"] = 1
    ###data
    if Dataset_N==1:
        profile =pd.read_csv("Species_Profile_full.csv",sep=',',low_memory=False) #read file with compositional data
        species=profile.to_numpy()[:,1:]/100 #normilized to 1
        full_list_bacteria = list(profile.columns)[1:]
        Where = lambda x: np.where(np.array(full_list_bacteria) == x)[0][0]
        arg_bacteria = [Where("Veillonella_parvula"), Where("Haemophilus_parainfluenzae")]
    if Dataset_N==2:
        profile =pd.read_csv("feature-table-not-filtered-l6.csv",sep=',',low_memory=False)
        profile.index = profile['Sample_id']
        profile = profile.drop(["Unassigned;__;__;__;__;__", "k__Archaea;__;__;__;__;__", "k__Bacteria;__;__;__;__;__"])
        profile = profile.drop(["Sample_id"], axis=1)
        profile = profile.transpose()
        species=profile.to_numpy()[:,1:]
        species = normalize(species, axis=1, norm='l1')
    d_full = np.shape(species)[1]
    ###prepocessing
    if Dataset_N==1:
        X_full, X_0, X_1, labels, labels_0, labels_1 = [[] for _ in range(6)] #X is data in 2 rows, labels are EPDS at tp 2
        ID = profile.Sample_id.to_numpy() #id of a sample in profile
        Individual_ID = Metadata.Individual_ID #id of a person
    if Dataset_N==2:
        X_full, X_01, X_02, X_12, labels, labels_01, labels_02, labels_12 = [[] for _ in range(8)] #X is data in 3 rows, labels are AP case at any tp
        ID = profile.index #id of a sample in profile
        Individual_ID = Metadata.family_id.to_numpy() #id of a person
        for k in Metadata.index:
            j = Metadata.tp[k]
            if j in [0,1,2]:
                i = Metadata.family_id[k]
                tp_id = "sub." + str(i) + time[j]
                if 1 - (tp_id in ID):
                    Metadata.loc[k, "tp"] = np.nan #checking if microbiome is available
    #dividing data into complete and incomplete
    if Dataset_N==1:
        for i in np.unique(Individual_ID):
            TP = Metadata.TimePoint[Individual_ID == i].to_numpy()
            if sum((TP - np.array([0,1,2]))**2)!=0:
                warnings.warn("TP are missing or in incorrect order") #does not happen in the dataset we have
            Outcomes = TARGET[Individual_ID == i]
            if Outcomes[2]!=2:
                if Imputation_type == "Oversampling" and Outcomes[2]==0:
                    oversampling = 0
                else:
                    oversampling = 1
                Reads = Metadata.ReadsNumber[Individual_ID == i].to_numpy()
                Sample_ID = Metadata.Sample_ID[Individual_ID == i] #id of a sample in metadata
                if 1-np.isnan(Reads[0]) and 1-np.isnan(Reads[1]): #complete data
                    labels = np.append(labels, Outcomes[2])
                    Dataset_line = np.zeros((1,2,d_full))
                    for j in range(2):
                        tp_id = Sample_ID[TP == j].to_numpy()
                        Dataset_line[:, j, :] = species[np.where(ID == tp_id)[0], :]
                    X_full = Dataset_line if np.size(X_full) == 0 else np.concatenate([X_full, Dataset_line])
                if 1-np.isnan(Reads[0]) and np.isnan(Reads[1]) and oversampling == 1: #tp 1 is missing
                    labels_0 = np.append(labels_0, Outcomes[2])
                    Dataset_line = np.zeros((1,2,d_full))
                    tp_id = Sample_ID[TP == 0].to_numpy()
                    Dataset_line[:, 0, :] = species[np.where(ID == tp_id)[0], :]
                    X_0 = Dataset_line if np.size(X_0) == 0 else np.concatenate([X_0, Dataset_line])
                if np.isnan(Reads[0]) and 1-np.isnan(Reads[1]) and oversampling == 1: #tp 0 is missing
                    labels_1 = np.append(labels_1, Outcomes[2])
                    Dataset_line = np.zeros((1,2,d_full))
                    tp_id = Sample_ID[TP == 1].to_numpy()
                    Dataset_line[:, 1, :] = species[np.where(ID == tp_id)[0], :]
                    X_1 = Dataset_line if np.size(X_1) == 0 else np.concatenate([X_1, Dataset_line])
    if Dataset_N==2:
        for i in np.unique(Individual_ID):
            TP = Metadata.tp[Individual_ID == i].to_numpy()
            Outcomes = TARGET[Individual_ID == i]
            if np.mean(Outcomes)==Outcomes[-1]: #considering data points with unique tp
                if Imputation_type == "Oversampling" and Outcomes[2]==0:
                    oversampling = 0
                else:
                    oversampling = 1
                if np.sum(TP == 0) == 1 and np.sum(TP == 1) == 1 and np.sum(TP == 2) == 1:
                    labels = np.append(labels, Outcomes[-1])
                    Dataset_line = np.zeros((1, 3, d_full))
                    for j in [0, 1, 2]:
                        tp_id = "sub." + str(i) + time[j]
                        Dataset_line[:, j, :] = species[np.where(ID == tp_id)[0], :]
                    X_full = Dataset_line if np.size(X_full) == 0 else np.concatenate([X_full, Dataset_line])
                if np.sum(TP == 0) == 1 and np.sum(TP == 1) == 1 and np.sum(TP == 2) == 0 and oversampling == 1:
                    labels_01 = np.append(labels_01, Outcomes[-1])
                    Dataset_line = np.zeros((1, 3, d_full))
                    for j in [0, 1]:
                        tp_id = "sub." + str(i) + time[j]
                        Dataset_line[:, j, :] = species[np.where(ID == tp_id)[0], :]
                    X_01 = Dataset_line if np.size(X_01) == 0 else np.concatenate([X_01, Dataset_line])
                if np.sum(TP == 0) == 1 and np.sum(TP == 1) == 0 and np.sum(TP == 2) == 1 and oversampling == 1:
                    labels_02 = np.append(labels_02, Outcomes[-1])
                    Dataset_line = np.zeros((1, 3, d_full))
                    for j in [0, 2]:
                        tp_id = "sub." + str(i) + time[j]
                        Dataset_line[:, j, :] = species[np.where(ID == tp_id)[0], :]
                    X_02 = Dataset_line if np.size(X_02) == 0 else np.concatenate([X_02, Dataset_line])
                if np.sum(TP == 0) == 0 and np.sum(TP == 1) == 1 and np.sum(TP == 2) == 1 and oversampling == 1:
                    labels_12 = np.append(labels_12, Outcomes[-1])
                    Dataset_line = np.zeros((1, 3, d_full))
                    for j in [1, 2]:
                        tp_id = "sub." + str(i) + time[j]
                        Dataset_line[:, j, :] = species[np.where(ID == tp_id)[0], :]
                    X_12 = Dataset_line if np.size(X_12) == 0 else np.concatenate([X_12, Dataset_line])
    X_f, X_t = X_full[labels == 0], X_full[labels == 1] #f = healthy, t = depressed
    size_f, size_t = np.shape(X_f)[0], np.shape(X_t)[0]
    print("sizes of complete f and t sets:",size_f,size_t)
    test_f, test_t = int(size_f*test), int(size_t*test) #amounts to put to test set
    print("sizes of test sets",test_f, test_t)
    AUC_TEST, N_OPT, SENSITIVITY, SPECIFICITY = [[] for _ in range(4)]
    for IS in range(Imputation_sets):
        sample_seed = IS + ITER
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
                if Dataset_N == 1:
                    X_test_cut, X_0_cut, X_1_cut = X_test, X_0, X_1
                    X = [X_train, X_val, X_test_cut, X_0_cut, X_1_cut]
                    d = np.shape(X[0])[2]
                    delta = np.min(X[0][X[0] > 0])
                    print("delta", delta)  # the smallest element
                    for xi in range(len(X)):
                        X[xi] = (X[xi] + delta) / (1 + delta * d) #remove 0s
                    X_train, X_val, X_test_cut, X_0_cut, X_1_cut = X
                    if Imputation_type!="No":
                        ###classes
                        classes_train, classes_val, classes_test = np.ones((np.shape(X_train)[0],1)),np.ones((np.shape(X_val)[0],1)),np.ones((np.shape(X_test_cut)[0],1))
                        classes_0 = np.zeros((np.shape(X_0_cut)[0], 1))
                        classes_1 = np.zeros((np.shape(X_1_cut)[0], 1))
                        ###imputation
                        X_1_cut = Imputation_library.impute(X_train, X_val, X_1_cut, 1, 0, sample_seed, arg_bacteria, Imputation_by)
                        X_0_cut = Imputation_library.impute(X_train, X_val, X_0_cut, 0, 1, sample_seed, arg_bacteria, Imputation_by)
                        X_train = np.concatenate([X_train, X_0_cut, X_1_cut]) ### concatenate full set with imputed sets
                        classes_train = np.concatenate([classes_train, classes_0,classes_1])
                        label_train = np.concatenate([label_train, labels_0, labels_1])
                    stats_train, stats_val = Transformation_library.transform_2(X_train,  arg_bacteria), Transformation_library.transform_2(X_val,  arg_bacteria)### construct features from 2 bacteria
                if Dataset_N == 2:
                    X_test_cut, X_01_cut, X_02_cut, X_12_cut = X_test, X_01, X_02, X_12 #no changes in data dimensionality if is_sparse = False
                    X = [X_train, X_val, X_test_cut, X_01_cut, X_02_cut, X_12_cut]
                    if is_sparse == 1:
                        if Imputation_type=="No":
                            X_combine = X_train.reshape(-1,d_full)
                        else:
                            X_combine = np.concatenate([X_train.reshape(-1,d_full), X_01_cut[:,[0,1],:].reshape(-1,d_full),X_02_cut[:,[0,2],:].reshape(-1,d_full),X_12_cut[:,[1,2],:].reshape(-1,d_full)])
                        sparsity = np.sum(X_combine == 0, axis=0) / np.shape(X_combine)[0]
                        for xi in range(len(X)):
                            other_species = np.sum(X[xi][:, :, sparsity >= sparsity_level], axis=-1).reshape(-1, 3, 1)
                            X[xi] = X[xi][:, :, sparsity < sparsity_level]
                            X[xi]= np.concatenate([X[xi], other_species], axis=-1)
                    d = np.shape(X[0])[2]
                    print("d=", d)
                    delta = np.min(X[0][X[0] > 0])
                    print("delta", delta)  # the smallest element
                    for xi in range(len(X)):
                        X[xi] = (X[xi] + delta) / (1 + delta * d) #remove 0s
                    X_train, X_val, X_test_cut, X_01_cut, X_02_cut, X_12_cut = X
                    if Imputation_type!="No":
                        ###classes
                        classes_train, classes_val, classes_test = np.ones((np.shape(X_train)[0], 3)), np.ones((np.shape(X_val)[0], 3)), np.ones((np.shape(X_test_cut)[0], 3))
                        classes_01, classes_02, classes_12 = np.ones((np.shape(X_01_cut)[0], 3)), np.ones((np.shape(X_02_cut)[0], 3)), np.ones((np.shape(X_12_cut)[0], 3))
                        classes_01[:,2], classes_02[:,1], classes_12[:,0] = 0,0,0
                        ###imputation
                        X_01_cut = Imputation_library.impute_one(X_train, X_val, X_01_cut, [0, 1], 2, sample_seed, Imputation_by)
                        X_02_cut = Imputation_library.impute_one(X_train, X_val, X_02_cut, [0, 2], 1, sample_seed, Imputation_by)
                        X_12_cut = Imputation_library.impute_one(X_train, X_val, X_12_cut, [1, 2], 0, sample_seed, Imputation_by)
                        X_train = np.concatenate([X_train, X_01_cut, X_02_cut, X_12_cut])
                        classes_train = np.concatenate([classes_train, classes_01, classes_02, classes_12])
                        label_train = np.concatenate([label_train, labels_01, labels_02, labels_12])
                    dY = 3 * d
                    X_train, X_val = np.reshape(X_train, (-1, dY)) / 3, np.reshape(X_val, (-1, dY)) / 3  # flatten inputs to one compositional vector
                    ### feature extraction
                    if Transoformation_type == "CLR":
                        stats_train, stats_val = Transformation_library.CLR(X_train), Transformation_library.CLR(X_val)
                    elif Transoformation_type == "ALR":
                        stats_train, stats_val = Transformation_library.LR(X_train), Transformation_library.LR(X_val)
                    elif Transoformation_type == "PLR":
                        ### construct hierarchy tree
                        Y_train = Transformation_library.CLR(X_train)
                        Z = hierarchy.linkage(np.transpose(Y_train), method="ward")
                        clustersize, clusters = Transformation_library.hierarchy_tree(Z, dY)
                        stats_train, stats_val = Transformation_library.data_transformation(X_train, Z, clustersize, clusters,dY), Transformation_library.data_transformation(X_val, Z, clustersize, clusters, dY)  ### construct features from hierarchy tree
                    else:
                        stats_train, stats_val = X_train, X_val
                if Imputation_type=="Imputation":
                    stats_train, stats_val = np.concatenate([classes_train, stats_train], axis=-1), np.concatenate([classes_val, stats_val], axis=-1)
                n_features = np.min([n_features, np.shape(stats_train)[1]])
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
            X_rest, label_rest = np.concatenate([X_train, X_val]), np.concatenate([label_train,label_val]) #concatenate everything except testing set
            if Dataset_N==1:
                stats_rest, stats_test = Transformation_library.transform_2(X_rest,  arg_bacteria), Transformation_library.transform_2(X_test_cut,  arg_bacteria)### construct features from 2 bacteria
            if Dataset_N==2:
                X_test_cut = np.reshape(X_test_cut, (-1, dY)) / 3
                if Transoformation_type == "CLR":
                    stats_rest, stats_test = Transformation_library.CLR(X_rest), Transformation_library.CLR(X_test_cut)
                elif Transoformation_type == "ALR":
                    stats_rest, stats_test = Transformation_library.LR(X_rest), Transformation_library.LR(X_test_cut)
                elif Transoformation_type == "PLR":
                    ### construct hierarchy tree
                    Y_rest = Transformation_library.CLR(X_rest)
                    Z = hierarchy.linkage(np.transpose(Y_rest), method="ward")
                    clustersize, clusters = Transformation_library.hierarchy_tree(Z, dY)
                    stats_rest, stats_test = Transformation_library.data_transformation(X_rest, Z, clustersize, clusters,dY), Transformation_library.data_transformation(X_test_cut, Z, clustersize, clusters, dY)  ### construct features from hierarchy tree
                else:
                    stats_rest, stats_test = X_rest, X_test_cut
            if Imputation_type=="Imputation":
                classes_rest = np.concatenate([classes_train,classes_val]) #concatenate everything except testing set
                stats_rest, stats_test = np.concatenate([classes_rest, stats_rest], axis= -1), np.concatenate([classes_test, stats_test], axis= -1)
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
    print(np.std(AUC_TEST))
    result= stats.ttest_1samp(AUC_TEST, level, alternative="greater")
    print("p value",result.pvalue)
    print("true positive",np.mean(SENSITIVITY))
    print("true negative",np.mean(SPECIFICITY))
    print("n opt",np.mean(N_OPT))
if __name__ == "__main__":
    #is_balanced = 0 #do we want to use weights during classification?
    #is_median = 1 #do we want to use median or mean when select optimal n (number of features)
    #Dataset_N = 1 #number of the dataset
    #Imputation_type = ["No", "Imputation", "Oversampling"][1]
    #Imputation_by = ["linear", "SVR", "GPR", "CVAE", "cGAN"][2]  #if imputation (or oversampling) is applied
    #Transoformation_type = ["Compositional","CLR", "ALR", "PLR"][2] #For Dataset 2, because the features are fixed for Dataset 1
    parser = argparse.ArgumentParser(description="Script for Transformation, Imputation, and Forecasting")
    parser.add_argument("--is_balanced", required=False, type=int, default=0,
                        help="class weight fpr random forest classifier. 0 is default, 1 is for balancing samples.")
    parser.add_argument("--is_median", required=False, type=int, default=0, help="1 if select optimal number of features ny median, 0 otherwise")
    parser.add_argument("--Dataset_N", required=False, type=int, default=2,
                        help="1 is for Dataset of depression, 2 is for Dataset of Food allergy")
    parser.add_argument("--Imputation_type", required=False, type=str, default="Imputation",
                        help="Do you want to complement data with artificial values? The choice is from 'No, Imputation, Oversampling', where Oversampling stands for including only positive labels")
    parser.add_argument("--Imputation_by", required=False, type=str, default="cGAN",
                        help="linear, SVR, GPR,CVAE, or cGAN. If imputation is used, what type of model to use")
    parser.add_argument("--Transoformation_type", required=False, type=str, default="Compositional",
                        help="Compositional, CLR, ALR, or PLR. For Dataset 2, the choice of log-transformation.")
    args = parser.parse_args()
    is_balanced = args.is_balanced
    is_median = args.is_median
    Dataset_N = args.Dataset_N
    Imputation_type = args.Imputation_type
    Imputation_by = args.Imputation_by
    Transoformation_type = args.Transoformation_type
    script(is_balanced, is_median, Dataset_N, Imputation_type, Imputation_by, Transoformation_type)