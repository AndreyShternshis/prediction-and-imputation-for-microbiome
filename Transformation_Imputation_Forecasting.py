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
###import libraries for log-transformation and imputation
import sys
sys.path.insert(0, '/libraries')
from libraries import Transformation_library, Imputation_library
###fix random seed for reproducibility 
import os
os.environ["PYTHONHASHSEED"] = "0"
import argparse
def rubins_rules(accuracies, n):
    #correction of variance by Rubin's rule with multiple imputations
    accuracies = accuracies.reshape(-1, n)
    estimates = np.mean(accuracies, axis=1)
    variances = np.var(accuracies, axis=1)
    M = len(estimates)
    Q_bar = np.mean(estimates)
    U_bar = np.mean(variances)
    B = np.sum((estimates - Q_bar) ** 2) / (M - 1)
    T = U_bar + (1 + 1 / M) * B
    SE = np.sqrt(T)
    z = 1.96
    r = (1 + 1 / M) * B / U_bar
    nu = (M - 1) * (1 + r ** (-1)) ** 2
    return Q_bar, T, Q_bar - z * SE, Q_bar + z * SE, nu
def script(is_balanced, is_median, Dataset_N, Imputation_type, Imputation_by, Transoformation_type):
    ###standard parameters
    level = 0.5  #balanced accuracy when predictions are made by random guess
    test = 0.2  #fraction of data used for test set
    val = 0.2  # fraction for validation (from the rest part)
    ITER = 5  # number of iteration for random forest (section 3.5)
    n_features = 20  #the maximum number of features when selecting by ANOVA (as discussed in section 3.2)
    test_sets = int(1 / test)  # number of testing sets (5)
    val_sets = int(1 / val)  # number of validation sets (5)
    if Dataset_N == 1:
        binary = 11  #threshold to be classified as helthy/depressed 
    if Dataset_N == 2:
        is_sparse = 1  # do we want to reduce sparsity?
        sparsity_level = 0.9  # sparsity threshold to merge bacteria together
    if is_balanced == 1:
        class_weight = "balanced"
    else:
        class_weight = None
    if Imputation_type == "No" or Imputation_by in ["linear", "SVR"]:  # number of imputation sets
        Imputation_sets = 1
    else:
        Imputation_sets = 5
    #read metadata file with id of participants and time points (depending on dataset number)
    if Dataset_N == 1:
        Metadata = pd.read_csv("BASIC_metadata_full.csv", sep=',',
                               low_memory=False)  # read file with depression levels and ids of participants
        Metadata.loc[Metadata.TimePoint == "Trimester2", "TimePoint"] = 0  # timepoits are 0,1,2
        Metadata.loc[Metadata.TimePoint == "Trimester3", "TimePoint"] = 1
        Metadata.loc[Metadata.TimePoint == "PostpartumWeek6", "TimePoint"] = 2
        i = Metadata[Metadata.ReadsNumber < 500000].index  # remove insufficient reads
        TARGET = 2 * np.ones_like(Metadata.EPDS)  # 2 is for missingness
        TARGET[Metadata.EPDS > binary] = 1  # binary labels for depression
        TARGET[Metadata.EPDS <= binary] = 0
    if Dataset_N == 2:
        Metadata = pd.read_csv("GMAP_metadata_public.csv", sep=",", low_memory=False)
        Metadata = Metadata.rename(columns={"sample": "tp"})
        Metadata.loc[Metadata.tp == "3_twomonth", "tp"] = 0
        Metadata.loc[Metadata.tp == "4_fourmonth", "tp"] = 1
        Metadata.loc[Metadata.tp == "5_sixmonth", "tp"] = 2
        time = [".3.twomonth", ".4.fourmonth", ".5.sixmonth"]
        TARGET = np.zeros_like(Metadata.case_id)
        TARGET[Metadata.case_id.to_numpy() == "AP Case"] = 1
    #read file with compositional data (depending on dataset number)
    if Dataset_N == 1:
        profile = pd.read_csv("Species_Profile_full.csv", sep=',',
                              low_memory=False)  # read file with compositional data
        species = profile.to_numpy()[:, 1:] / 100  # normalized to 1
        full_list_bacteria = list(profile.columns)[1:]
        Where = lambda x: np.where(np.array(full_list_bacteria) == x)[0][0]
        arg_bacteria = [Where("Veillonella_parvula"), Where("Haemophilus_parainfluenzae")] #for Dataset 1 the bacteria space is predefined by species identified in Table 5
    if Dataset_N == 2:
        profile = pd.read_csv("feature-table-not-filtered-l6.csv", sep=',', low_memory=False)
        profile.index = profile['Sample_id']
        profile = profile.drop(["Unassigned;__;__;__;__;__", "k__Archaea;__;__;__;__;__", "k__Bacteria;__;__;__;__;__"])  #not annonated bacteria are ommited
        profile = profile.drop(["Sample_id"], axis=1)
        profile = profile.transpose()
        species = profile.to_numpy()[:, 1:]
        species = normalize(species, axis=1, norm='l1') #make the data compositional
    d_full = np.shape(species)[1]
    ###make separate arrays for complete and incomplete data points (depending on dataset number)
    if Dataset_N == 1:
        X_full, X_0, X_1, labels, labels_0, labels_1 = [[] for _ in
                                                        range(6)]  # X is data in 2 rows, labels are EPDS at tp 2
        ID = profile.Sample_id.to_numpy()  # id of a sample in profile
        Individual_ID = Metadata.Individual_ID  # id of a person
    if Dataset_N == 2:
        X_full, X_01, X_02, X_12, labels, labels_01, labels_02, labels_12 = [[] for _ in range(
            8)]  # X is data in 3 rows, labels are AP case at any tp
        ID = profile.index  # id of a sample in profile
        Individual_ID = Metadata.family_id.to_numpy()  # id of a person
        for k in Metadata.index:
            j = Metadata.tp[k]
            if j in [0, 1, 2]: #iterating in 3 time points
                i = Metadata.family_id[k]
                tp_id = "sub." + str(i) + time[j]
                if 1 - (tp_id in ID):
                    Metadata.loc[k, "tp"] = np.nan  # checking if microbiome is available
    #fill in the arrays for complete and  incomplete datapoints (depending on dataset number)
    if Dataset_N == 1:
        for i in np.unique(Individual_ID):
            TP = Metadata.TimePoint[Individual_ID == i].to_numpy()
            if sum((TP - np.array([0, 1, 2])) ** 2) != 0:
                warnings.warn("TP are missing or in incorrect order")  # does not happen in the dataset we have
            Outcomes = TARGET[Individual_ID == i]
            if Outcomes[2] != 2:
                if Imputation_type == "Oversampling" and Outcomes[2] == 0:
                    oversampling = 0
                else:
                    oversampling = 1
                Reads = Metadata.ReadsNumber[Individual_ID == i].to_numpy()
                Sample_ID = Metadata.Sample_ID[Individual_ID == i]  # id of a sample in metadata
                if 1 - np.isnan(Reads[0]) and 1 - np.isnan(Reads[1]):  # complete data
                    labels = np.append(labels, Outcomes[2])
                    Dataset_line = np.zeros((1, 2, d_full))
                    for j in range(2):
                        tp_id = Sample_ID[TP == j].to_numpy()
                        Dataset_line[:, j, :] = species[np.where(ID == tp_id)[0], :]
                    X_full = Dataset_line if np.size(X_full) == 0 else np.concatenate([X_full, Dataset_line])
                if 1 - np.isnan(Reads[0]) and np.isnan(Reads[1]) and oversampling == 1:  # tp 1 is missing
                    labels_0 = np.append(labels_0, Outcomes[2])
                    Dataset_line = np.zeros((1, 2, d_full))
                    tp_id = Sample_ID[TP == 0].to_numpy()
                    Dataset_line[:, 0, :] = species[np.where(ID == tp_id)[0], :]
                    X_0 = Dataset_line if np.size(X_0) == 0 else np.concatenate([X_0, Dataset_line])
                if np.isnan(Reads[0]) and 1 - np.isnan(Reads[1]) and oversampling == 1:  # tp 0 is missing
                    labels_1 = np.append(labels_1, Outcomes[2])
                    Dataset_line = np.zeros((1, 2, d_full))
                    tp_id = Sample_ID[TP == 1].to_numpy()
                    Dataset_line[:, 1, :] = species[np.where(ID == tp_id)[0], :]
                    X_1 = Dataset_line if np.size(X_1) == 0 else np.concatenate([X_1, Dataset_line])
    if Dataset_N == 2:
        for i in np.unique(Individual_ID):
            TP = Metadata.tp[Individual_ID == i].to_numpy()
            Outcomes = TARGET[Individual_ID == i]
            if np.mean(Outcomes) == Outcomes[-1]:  # considering data points with unique tp
                if Imputation_type == "Oversampling" and Outcomes[2] == 0:
                    oversampling = 0
                else:
                    oversampling = 1
                if np.sum(TP == 0) == 1 and np.sum(TP == 1) == 1 and np.sum(TP == 2) == 1: #complete data
                    labels = np.append(labels, Outcomes[-1])
                    Dataset_line = np.zeros((1, 3, d_full))
                    for j in [0, 1, 2]:
                        tp_id = "sub." + str(i) + time[j]
                        Dataset_line[:, j, :] = species[np.where(ID == tp_id)[0], :]
                    X_full = Dataset_line if np.size(X_full) == 0 else np.concatenate([X_full, Dataset_line])
                if np.sum(TP == 0) == 1 and np.sum(TP == 1) == 1 and np.sum(TP == 2) == 0 and oversampling == 1: #tp2 is missing
                    labels_01 = np.append(labels_01, Outcomes[-1])
                    Dataset_line = np.zeros((1, 3, d_full))
                    for j in [0, 1]:
                        tp_id = "sub." + str(i) + time[j]
                        Dataset_line[:, j, :] = species[np.where(ID == tp_id)[0], :]
                    X_01 = Dataset_line if np.size(X_01) == 0 else np.concatenate([X_01, Dataset_line])
                if np.sum(TP == 0) == 1 and np.sum(TP == 1) == 0 and np.sum(TP == 2) == 1 and oversampling == 1: #tp1 is missing
                    labels_02 = np.append(labels_02, Outcomes[-1])
                    Dataset_line = np.zeros((1, 3, d_full))
                    for j in [0, 2]:
                        tp_id = "sub." + str(i) + time[j]
                        Dataset_line[:, j, :] = species[np.where(ID == tp_id)[0], :]
                    X_02 = Dataset_line if np.size(X_02) == 0 else np.concatenate([X_02, Dataset_line])
                if np.sum(TP == 0) == 0 and np.sum(TP == 1) == 1 and np.sum(TP == 2) == 1 and oversampling == 1: #tp0 is missing
                    labels_12 = np.append(labels_12, Outcomes[-1])
                    Dataset_line = np.zeros((1, 3, d_full))
                    for j in [1, 2]:
                        tp_id = "sub." + str(i) + time[j]
                        Dataset_line[:, j, :] = species[np.where(ID == tp_id)[0], :]
                    X_12 = Dataset_line if np.size(X_12) == 0 else np.concatenate([X_12, Dataset_line])
    ###imputation, transformation, and classification of data into for-loops for imputation, test, and validation sets
    X_f, X_t = X_full[labels == 0], X_full[labels == 1]  # f = healthy, t = depressed/allegic
    size_f, size_t = np.shape(X_f)[0], np.shape(X_t)[0]
    print("sizes of complete f and t sets:", size_f, size_t)
    test_f, test_t = int(size_f * test), int(size_t * test)  # sizes of test sets
    AUC_TEST, N_OPT, SENSITIVITY, SPECIFICITY = [[] for _ in range(4)]
    for IS in range(Imputation_sets): #iterating over imputation sets since the imputed values can be random
        sample_seed = IS + ITER
        aucs_val = np.zeros((val_sets, ITER, n_features))
        for TS in range(test_sets):  #iterating over test sets
            print("test group", TS)
            test_f_range, test_t_range = range(TS * test_f, (TS + 1) * test_f), range(TS * test_t, (TS + 1) * test_t)
            rest_f_range, rest_t_range = np.setdiff1d(range(size_f), test_f_range), np.setdiff1d(range(size_t),
                                                                                                 test_t_range)
            X_f_test, X_t_test = X_f[test_f_range], X_t[test_t_range]
            X_test = np.concatenate([X_f_test, X_t_test])
            label_test = np.zeros(test_f + test_t)
            label_test[test_f:] = 1
            rest_f, rest_t = np.size(rest_f_range), np.size(rest_t_range)
            val_f, val_t = int(rest_f * val), int(rest_t * val)  #sizes of validation sets
            train_f, train_t = rest_f - val_f, rest_t - val_t  # the rest of rest is for training set
            print("sizes of train sets", train_f, train_t)
            for VS in range(val_sets):  #iterating over vaidation sets
                print("val group", VS)
                val_f_range, val_t_range = rest_f_range[range(VS * val_f, (VS + 1) * val_f)], rest_t_range[
                    range(VS * val_t, (VS + 1) * val_t)]
                train_f_range, train_t_range = np.setdiff1d(rest_f_range, val_f_range), np.setdiff1d(rest_t_range,
                                                                                                     val_t_range)
                X_f_val, X_t_val, X_f_train, X_t_train = X_f[val_f_range], X_t[val_t_range], X_f[train_f_range], X_t[
                    train_t_range]
                X_val, X_train = np.concatenate([X_f_val, X_t_val]), np.concatenate([X_f_train, X_t_train])
                label_val, label_train = np.zeros(val_f + val_t), np.zeros(train_f + train_t)
                label_val[val_f:], label_train[train_f:] = 1, 1
                if Dataset_N == 1:
                    X_test_cut, X_0_cut, X_1_cut = X_test, X_0, X_1
                    X = [X_train, X_val, X_test_cut, X_0_cut, X_1_cut]
                    d = np.shape(X[0])[2]
                    delta = np.min(X[0][X[0] > 0]) # the smallest non-negative element used to replace 0 values (as discussed in section 3.3)
                    for xi in range(len(X)):
                        X[xi] = (X[xi] + delta) / (1 + delta * d)  # remove 0s
                    X_train, X_val, X_test_cut, X_0_cut, X_1_cut = X
                    if Imputation_type != "No":
                        ###Classes show what data were originally missing. They are the part of the input for classifier
                        classes_train, classes_val, classes_test = np.ones((np.shape(X_train)[0], 2)), np.ones(
                            (np.shape(X_val)[0], 2)), np.ones((np.shape(X_test_cut)[0], 2))
                        classes_0 = np.ones((np.shape(X_0_cut)[0], 2))
                        classes_1 = np.ones((np.shape(X_1_cut)[0], 2))
                        classes_0[:, 1] = 0
                        classes_1[:, 0] = 0
                        ###Imputation is done as descrbed in corresponding section 3.5
                        X_1_cut = Imputation_library.impute(X_train, X_val, X_1_cut, 1, 0, sample_seed, arg_bacteria,
                                                            Imputation_by)
                        X_0_cut = Imputation_library.impute(X_train, X_val, X_0_cut, 0, 1, sample_seed, arg_bacteria,
                                                            Imputation_by)
                        ### concatenate full set with imputed sets
                        X_train = np.concatenate(
                            [X_train, X_0_cut, X_1_cut]) 
                        classes_train = np.concatenate([classes_train, classes_0, classes_1])
                        label_train = np.concatenate([label_train, labels_0, labels_1])
                    ### construct features from 2 bacteria
                    stats_train, stats_val = Transformation_library.transform_2(X_train, arg_bacteria), Transformation_library.transform_2(X_val, arg_bacteria)
                if Dataset_N == 2:
                    ###bacteria are merged if decided by a user
                    X_test_cut, X_01_cut, X_02_cut, X_12_cut = X_test, X_01, X_02, X_12  # no changes in data dimensionality if is_sparse = False
                    X = [X_train, X_val, X_test_cut, X_01_cut, X_02_cut, X_12_cut]
                    if is_sparse == 1:
                        if Imputation_type == "No":
                            X_combine = X_train.reshape(-1, d_full)
                        else:
                            X_combine = np.concatenate(
                                [X_train.reshape(-1, d_full), X_01_cut[:, [0, 1], :].reshape(-1, d_full),
                                 X_02_cut[:, [0, 2], :].reshape(-1, d_full),
                                 X_12_cut[:, [1, 2], :].reshape(-1, d_full)])
                        sparsity = np.sum(X_combine == 0, axis=0) / np.shape(X_combine)[0]
                        for xi in range(len(X)):
                            other_species = np.sum(X[xi][:, :, sparsity >= sparsity_level], axis=-1).reshape(-1, 3, 1)
                            X[xi] = X[xi][:, :, sparsity < sparsity_level]
                            X[xi] = np.concatenate([X[xi], other_species], axis=-1)
                    d = np.shape(X[0])[2]
                    print("d=", d)
                    delta = np.min(X[0][X[0] > 0]) # the smallest non-negative element used to replace 0 values (as discussed in section 3.3)
                    for xi in range(len(X)):
                        X[xi] = (X[xi] + delta) / (1 + delta * d)  # remove 0s
                    X_train, X_val, X_test_cut, X_01_cut, X_02_cut, X_12_cut = X
                    if Imputation_type != "No":
                        ###classes
                        classes_train, classes_val, classes_test = np.ones((np.shape(X_train)[0], 3)), np.ones(
                            (np.shape(X_val)[0], 3)), np.ones((np.shape(X_test_cut)[0], 3))
                        classes_01, classes_02, classes_12 = np.ones((np.shape(X_01_cut)[0], 3)), np.ones(
                            (np.shape(X_02_cut)[0], 3)), np.ones((np.shape(X_12_cut)[0], 3))
                        classes_01[:, 2], classes_02[:, 1], classes_12[:, 0] = 0, 0, 0
                        ###imputation
                        X_01_cut = Imputation_library.impute_one(X_train, X_val, X_01_cut, [0, 1], 2, sample_seed,
                                                                 Imputation_by)
                        X_02_cut = Imputation_library.impute_one(X_train, X_val, X_02_cut, [0, 2], 1, sample_seed,
                                                                 Imputation_by)
                        X_12_cut = Imputation_library.impute_one(X_train, X_val, X_12_cut, [1, 2], 0, sample_seed,
                                                                 Imputation_by)
                        X_train = np.concatenate([X_train, X_01_cut, X_02_cut, X_12_cut])
                        classes_train = np.concatenate([classes_train, classes_01, classes_02, classes_12])
                        label_train = np.concatenate([label_train, labels_01, labels_02, labels_12])
                    dY = 3 * d #the dimensionality is 3 time points multipled by the number of bacteria
                    X_train, X_val = np.reshape(X_train, (-1, dY)) / 3, np.reshape(X_val, (-1,
                                                                                           dY)) / 3  # flatten inputs to one compositional vector
                    ### log-transformations from setion 3.3
                    if Transoformation_type == "CLR":
                        stats_train, stats_val = Transformation_library.CLR(X_train), Transformation_library.CLR(X_val)
                    elif Transoformation_type == "ALR":
                        stats_train, stats_val = Transformation_library.LR(X_train), Transformation_library.LR(X_val)
                    elif Transoformation_type == "PLR":
                        ### construct hierarchy tree as in Figure 1
                        Y_train = Transformation_library.CLR(X_train)
                        Z = hierarchy.linkage(np.transpose(Y_train), method="ward")
                        clustersize, clusters = Transformation_library.hierarchy_tree(Z, dY)
                        ### construct features from hierarchy tree
                        stats_train, stats_val = Transformation_library.data_transformation(X_train, Z, clustersize,
                                                                                            clusters,
                                                                                            dY), Transformation_library.data_transformation(
                            X_val, Z, clustersize, clusters, dY)  
                    else:
                        stats_train, stats_val = X_train, X_val
                if Imputation_type == "Imputation":
                    stats_train, stats_val = np.concatenate([classes_train, stats_train], axis=-1), np.concatenate(
                        [classes_val, stats_val], axis=-1)
                n_features = np.min([n_features, np.shape(stats_train)[1]])
                for j in range(n_features):  #choose the best j + 1 features using ANOVA (section 3.2)
                    sel = SelectKBest(f_classif, k=j + 1).fit(pd.DataFrame(stats_train), label_train)
                    Index = sel.get_support()
                    for seed in range(ITER):  #classify by random forest with different random seeds (section 3.4)
                        rf = RandomForestClassifier(random_state=seed, class_weight=class_weight).fit(
                            stats_train[:, Index], label_train)
                        y_pred = rf.predict(stats_val[:, Index])
                        aucs_val[VS, seed, j] = roc_auc_score(label_val, y_pred)
            if is_median == 1:  # optimal number of features by median of mean
                print("medians", np.median(aucs_val, axis=(0, 1)))  # average by validation sets and random seeds
                n_opt = np.argmax(np.median(aucs_val, axis=(0, 1))) + 1
            else:
                print("means", np.mean(aucs_val, axis=(0, 1)))
                n_opt = np.argmax(np.mean(aucs_val, axis=(0, 1))) + 1
            N_OPT = np.append(N_OPT, n_opt)
            X_rest, label_rest = np.concatenate([X_train, X_val]), np.concatenate(
                [label_train, label_val])  # concatenate data except testing set
            if Dataset_N == 1:
                ### construct features from 2 bacteria
                stats_rest, stats_test = Transformation_library.transform_2(X_rest,
                                                                            arg_bacteria), Transformation_library.transform_2(
                    X_test_cut, arg_bacteria) 
            if Dataset_N == 2:
                X_test_cut = np.reshape(X_test_cut, (-1, dY)) / 3 # flatten test samples to one compositional vector
                if Transoformation_type == "CLR":
                    stats_rest, stats_test = Transformation_library.CLR(X_rest), Transformation_library.CLR(X_test_cut)
                elif Transoformation_type == "ALR":
                    stats_rest, stats_test = Transformation_library.LR(X_rest), Transformation_library.LR(X_test_cut)
                elif Transoformation_type == "PLR":
                    ### construct hierarchy tree
                    Y_rest = Transformation_library.CLR(X_rest)
                    Z = hierarchy.linkage(np.transpose(Y_rest), method="ward")
                    clustersize, clusters = Transformation_library.hierarchy_tree(Z, dY)
                    ### construct features from hierarchy tree
                    stats_rest, stats_test = Transformation_library.data_transformation(X_rest, Z, clustersize,
                                                                                        clusters,
                                                                                        dY), Transformation_library.data_transformation(
                        X_test_cut, Z, clustersize, clusters, dY)  
                else:
                    stats_rest, stats_test = X_rest, X_test_cut
            if Imputation_type == "Imputation":
                classes_rest = np.concatenate([classes_train, classes_val])  # concatenate classes except testing set
                stats_rest, stats_test = np.concatenate([classes_rest, stats_rest], axis=-1), np.concatenate(
                    [classes_test, stats_test], axis=-1)
            sel = SelectKBest(f_classif, k=n_opt).fit(pd.DataFrame(stats_rest),
                                                      label_rest)  # select optimal number of features
            Index = sel.get_support()
            print("test features:",
                  sel.get_feature_names_out())  # from features names we can reconstruct what bacteria we use
            for seed in range(ITER): #iterations for random forest on test sets
                rf = RandomForestClassifier(random_state=seed, class_weight=class_weight).fit(stats_rest[:, Index],
                                                                                              label_rest)
                print("importances", rf.feature_importances_)  # display importances of features
                y_pred = rf.predict(stats_test[:, Index])
                print("auc", roc_auc_score(label_test, y_pred))  # display balanced accuracy
                AUC_TEST = np.append(AUC_TEST, roc_auc_score(label_test, y_pred))  # balanced accuracy on testing set
                SENSITIVITY = np.append(SENSITIVITY, sum(y_pred[label_test == 1] == 1) / sum(label_test == 1))
                SPECIFICITY = np.append(SPECIFICITY, sum(y_pred[label_test == 0] == 0) / sum(label_test == 0))
    #save data for the comparison between approaches 
    np.save(f"BA_{is_balanced}_{is_median}_{Dataset_N}_{Imputation_type}_{Imputation_by}_{Transoformation_type}",
            AUC_TEST)
    np.save(f"sens_{is_balanced}_{is_median}_{Dataset_N}_{Imputation_type}_{Imputation_by}_{Transoformation_type}",
            SENSITIVITY)
    np.save(f"spec_{is_balanced}_{is_median}_{Dataset_N}_{Imputation_type}_{Imputation_by}_{Transoformation_type}",
            SPECIFICITY)
    if Imputation_sets == 1:
        Q_bar, STD = np.mean(AUC_TEST), np.std(AUC_TEST)
        z = 1.96
        print(np.round(Q_bar, 2), np.round(STD, 2), np.round(Q_bar - z * STD, 2), np.round(Q_bar + z * STD, 2))
        result = stats.ttest_1samp(AUC_TEST, level, alternative="greater")
        print("p value", result.pvalue)
    else:
        Q_bar, T, LB, UB, nu = rubins_rules(AUC_TEST, ITER)
        print(np.round(Q_bar, 2), np.round(np.sqrt(T), 2), np.round(LB, 2), np.round(UB, 2))
        W = (Q_bar - level) / T
        print(W)
        pvalue = stats.f.sf(W, 1, nu)
        print("p value", pvalue)
    print("true positive", np.round(np.mean(SENSITIVITY), 2))
    print("true negative", np.round(np.mean(SPECIFICITY), 2))
    print("n opt", np.mean(N_OPT))
    print(N_OPT)
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