import numpy as np
import pandas as pd
import random
from datetime import datetime
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from igraph import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from time import time
import math
from itertools import chain

random.seed(42)


def read_data(dataset):
    df_edges = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    df_edges.columns = ['from', 'to']  # name the columns as from and to
    unique_nodes = ((df_edges['from']._append(df_edges['to'])).unique()).tolist()  # list all nodes in the dataset
    print("Graph edges are loaded")
    csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_indicator.txt",
                      header=None)  # returns a dataframe of graph indicators
    csv.columns = ["ID"]  # rename the column as ID
    graph_indicators = (csv["ID"].values.astype(int))  # convert the dataframe to array of integers
    print("Graph indicators are loaded")
    read_csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    read_csv.columns = ["ID"]
    graph_labels = (read_csv["ID"].values.astype(int))
    print("Graph labels are loaded")
    read_nodelabel = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_node_labels.txt", header=None)
    read_nodelabel.columns = ["ID"]
    node_labels = (read_nodelabel["ID"].values.astype(int))
    print("Node labels are loaded")

    unique_graph_indicator = np.arange(min(graph_indicators),
                                       max(graph_indicators) + 1)  # list unique graph ids

    return unique_graph_indicator, df_edges, graph_indicators, graph_labels, node_labels, unique_nodes


def read_labels(node_labels, unique_nodes):
    nodes_dict = {}
    for index, ele in enumerate(node_labels):
        idx = index + 1
        if idx in unique_nodes:
            nodes_is = {idx: ele}
        else:
            unique_nodes.append(
                idx)  # if index is not found as node_id, append the index as a new node and give its corresponding node label
            nodes_is = {idx: ele}  # generated a random no

        nodes_dict.update(nodes_is)  # appending the dictionary to the outer dictionary

    return nodes_dict


def operate_kernel(unique_graph_indicator, df_edges, graph_indicators, perc, graph_labels, iter_, dataset, node_labels):
    transform_data = []
    for graphid in unique_graph_indicator:
        graphid_loc = [index + 1 for index, element in enumerate(graph_indicators) if element == graphid]
        nodelabels = [ele for ind, ele in enumerate(node_labels, start=1) if ind in graphid_loc]
        edges_loc = df_edges[df_edges['from'].isin(graphid_loc)]  # return the edge list
        createGraph = Graph.TupleList(edges_loc.itertuples(index=False), directed=False, weights=True)
        nodedict = {i: nodelabels[i] for i, j in enumerate(graphid_loc)}

        if len(graphid_loc) > 2:
            deleteEdge = random.sample(createGraph.get_edgelist(),
                                       math.ceil(perc * (createGraph.ecount() / 2)))
            edges_loc_list = [edgeTuple for edgeTuple in createGraph.get_edgelist() if edgeTuple not in deleteEdge]
            nodes_edges = [edges_loc_list, nodedict]
        else:
            edges_loc_list = createGraph.get_edgelist()
            nodes_edges = [edges_loc_list, nodedict]

        transform_data.append(nodes_edges)

    g_train, g_test, y_train, y_test = train_test_split(transform_data, graph_labels, test_size=0.2)

    start2 = time()

    wl = WeisfeilerLehman(n_iter=iter_, base_graph_kernel=VertexHistogram, verbose=True, normalize=True)
    train_data = wl.fit_transform(g_train)
    test_data = wl.transform(g_test)

    t2 = time()
    time2 = t2 - start2

    return train_data, test_data, y_train, y_test, time2


def tuning_hyperparamters():
    n_estimators = [int(a) for a in np.linspace(start=200, stop=500, num=5)]
    max_depth = [int(b) for b in np.linspace(start=2, stop=10, num=6)]
    num_cv = 10
    bootstrap = [True, False]
    gridlength = len(n_estimators) * len(max_depth) * num_cv
    print(str(gridlength) + " RFs will be created in the grid search.")
    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)

    return param_grid, num_cv


def random_forest(train_data, test_data, y_train, y_test, param_grid, num_cv, time2, iter_, dataset, perc):
    start3 = time()
    print(dataset + " training started at", datetime.now().strftime("%H:%M:%S"))

    rfc = RandomForestClassifier()
    grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=num_cv, n_jobs=10)
    grid.fit(train_data, y_train)
    param_choose = grid.best_params_
    if len(set(y_test)) > 2:  # multiclass case
        print(dataset + " requires multi class RF")
        forest = RandomForestClassifier(**param_choose, verbose=1).fit(train_data, y_train)
        y_pred = forest.predict(test_data)
        y_preda = forest.predict_proba(test_data)
        auc = roc_auc_score(y_test, y_preda, multi_class="ovr")
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
    else:  # binary case
        rfc_pred = RandomForestClassifier(**param_choose, verbose=1).fit(train_data, y_train)
        test_pred = rfc_pred.predict(test_data)
        auc = roc_auc_score(y_test, rfc_pred.predict_proba(test_data)[:, 1])
        accuracy = accuracy_score(y_test, test_pred)
        conf_mat = confusion_matrix(y_test, test_pred)
    print(dataset + " accuracy is " + str(accuracy) + ", AUC is " + str(auc))

    t3 = time()
    time3 = t3 - start3

    print(f'Kernel took {time2} seconds, training took {time3} seconds')
    flat_conf_mat = (str(conf_mat.flatten(order='C')))[1:-1]
    file.write(
        f"{dataset}\t{str(perc)}\t{str(time2)}\t{str(time3)}\t{str(accuracy)}\t{str(auc)}\t{str(iter_)}\t{str(flat_conf_mat)}\n")

    file.flush()


def main1(dataset, iter_, perc):
    unique_graph_indicator, df_edges, graph_indicators, graph_labels, node_labels, unique_nodes = read_data(dataset)
    nodes_dict = read_labels(node_labels, unique_nodes)
    train_data, test_data, y_train, y_test, time2 = operate_kernel(unique_graph_indicator, df_edges, graph_indicators,
                                                                   perc, graph_labels, iter_, dataset, node_labels)

    return train_data, test_data, y_train, y_test, time2


def main2(train_data, test_data, y_train, y_test, time2, dataset, iter_, perc):
    param_grid, num_cv = tuning_hyperparamters()
    random_forest(train_data, test_data, y_train, y_test, param_grid, num_cv, time2, iter_, dataset, perc)


if __name__ == '__main__':
    data_path = "path to data"  # dataset path on computer
    data_list = "list of data used"
    outputfile = "save result to file"
    with open(outputfile, "w") as file:
        header = 'dataset\tperc.of removal\tkernelTime\ttrainTime\taccuracy\tauc\titer\tflat_conf_mat\n'
        file.write(header)
        for dataset in data_list:
            for iter in (2, 3):
                for per in [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
                    train_data, test_data, y_train, y_test, time2 = main1(dataset, iter_=iter, perc=per)
                    for duplication in np.arange(10):
                        main2(train_data, test_data, y_train, y_test, time2, dataset, iter_=iter, perc=per)