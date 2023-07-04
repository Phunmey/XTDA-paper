"""
created by: Taiwo Funmilola Mary
Date: 02-03-2023
description: read the landscape and silhouette csv files and train the randomforest
"""

import random
import sys
from datetime import datetime
from time import time
import numpy as np
import glob
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

random.seed(123)


def read_data(files):
    csv_tda = pd.read_csv(files, header=0)
    landscapeTime = None
    if len(csv_tda.columns) == 5:
        landscapeTime = csv_tda["landscapeTime"].sum()
        csv_tda["landscapeList"] = csv_tda["landscapeList"].str.strip('[]')  # remove [] from the list
        features = csv_tda["landscapeList"].str.split(",", n=-1, expand=True)  # split the column to multiple columns
    else:
        csv_tda["res_1"] = csv_tda["res_1"].str.strip('[]')  # remove [] from the list
        feature1 = csv_tda["res_1"].str.split(",", n=-1, expand=True)
        feature2 = csv_tda.iloc[:, 4:]
        features = pd.concat([feature1, feature2], axis=1, ignore_index=True)
        features = features.dropna(axis=1)
    labels = csv_tda["graphLabel"]
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    return x_train, x_test, y_train, y_test, landscapeTime


def tuning_hyperparameter():
    n_estimators = [int(a) for a in np.linspace(start=200, stop=500, num=5)]
    max_depth = [int(b) for b in np.linspace(start=2, stop=10, num=6)]
    num_cv = 10
    bootstrap = [True, False]
    gridlength = len(n_estimators) * len(max_depth) * num_cv
    print(str(gridlength) + " RFs will be created in the grid search.")
    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)

    return param_grid, num_cv


def random_forest(filename, param_grid, x_train, x_test, y_train, y_test, num_cv, landscapeTime):
    print(filename + " training started at", datetime.now().strftime("%H:%M:%S"))
    start = time()

    rfc = RandomForestClassifier()
    grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=num_cv, n_jobs=10)
    grid.fit(x_train, y_train)
    param_choose = grid.best_params_
    if len(set(y_test)) > 2:  # multiclass case
        print(filename + " requires multi class RF")
        forest = RandomForestClassifier(**param_choose, verbose=1).fit(x_train, y_train)
        y_pred = forest.predict(x_test)
        y_preda = forest.predict_proba(x_test)
        auc = roc_auc_score(y_test, y_preda, multi_class="ovr")
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
    else:  # binary case
        rfc_pred = RandomForestClassifier(**param_choose, verbose=1).fit(x_train, y_train)
        test_pred = rfc_pred.predict(x_test)
        auc = roc_auc_score(y_test, rfc_pred.predict_proba(x_test)[:, 1])
        accuracy = accuracy_score(y_test, test_pred)
        conf_mat = confusion_matrix(y_test, test_pred)
    print(filename + " accuracy is " + str(accuracy) + ", AUC is " + str(auc))

    t = time()
    training_time = t - start

    print(f'rips complex training took {training_time} seconds')
    flat_conf_mat = (str(conf_mat.flatten(order='C')))[
                    1:-1]  # flatten confusion matrix into a single row while removing the [ ]
    file.write(filename + "\t" + str(training_time) + "\t" + str(landscapeTime) + "\t" + str(accuracy) + "\t" + str(auc) + "\t" + str(
        flat_conf_mat) + "\n")

    file.flush()


def main():
    x_train, x_test, y_train, y_test, landscapeTime = read_data(files)
    param_grid, num_cv = tuning_hyperparameter()
    random_forest(filename, param_grid, x_train, x_test, y_train, y_test, num_cv, landscapeTime)


if __name__ == "__main__":
    data_path = "/home/taiwo/projects/def-cakcora/taiwo/Apr2023/result/landscapeSilhouette/featureDataLandscape"  # dataset path on computer
    collect_files = os.path.join(data_path + "/*.csv")  # merging the files
    list_files = glob.glob(collect_files)  # A list of all collected files is returned
    outputFile = "/home/taiwo/projects/def-cakcora/taiwo/Apr2023/result/randomForest/" + 'rfLandscape.csv'
    file = open(outputFile, 'w')
    for files in list_files:
        filename = os.path.basename(files).split(".")[0]  # filename without the extension
        for duplication in np.arange(10):
            main()

    file.close()
