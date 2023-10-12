import random
from datetime import datetime
from time import time
import numpy as np
import pandas as pd
from igraph import *
from ripser import ripser

random.seed(42)


def reading_csv(dataset):
    df_edges = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    df_edges.columns = ['from', 'to']
    print("Graph edges are loaded")
    csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_indicator.txt", header=None)
    print("Graph indicators are loaded")
    csv.columns = ["ID"]
    graph_indicators = (csv["ID"].values.astype(int))
    read_csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    read_csv.columns = ["ID"]
    graph_labels = (read_csv["ID"].values.astype(int))
    print("Graph labels are loaded")
    unique_graph_indicator = np.arange(min(graph_indicators),
                                       max(graph_indicators) + 1)  # list unique graph ids in a dataset

    return unique_graph_indicator, graph_labels, df_edges, graph_indicators


def alpha_filt(unique_graph_indicator, graph_indicators, df_edges, step_size, dataset,
               graph_labels):  # this is for the train data

    betti_list = []
    density = []
    diameter = []
    clustering_coeff = []
    spectral_gap = []
    assortativity_ = []
    cliques = []
    motifs = []
    components = []

    for graph_id in unique_graph_indicator:
        id_location = [index + 1 for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_label = [ele for ind, ele in enumerate(graph_labels, start=1) if
                       ind == graph_id]  # obtain graph label corresponding to the graph_id
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
        create_traingraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
        create_dmatrix = np.asarray(Graph.shortest_paths_dijkstra(create_traingraph))
        norm_dmatrix = create_dmatrix / np.nanmax(create_dmatrix[create_dmatrix != np.inf])

        start = time()
        train_rips = ripser(norm_dmatrix, thresh=1, maxdim=1, distance_matrix=True)[
            'dgms']
        filtr_time = time() - start

        #    splitting the dimension into 0 and 1
        train_dgm_0 = train_rips[0]
        train_dgm_1 = train_rips[1]

        # save the persistence diagrams
        filename = "save PD"
        with open(filename, "a") as f:
            f.write(f"{graph_id}: {train_dgm_0, train_dgm_1}\n")
            f.flush()  # ensures data is written to file
            f.close()

        #    obtain betti numbers for the unique dimensions
        train_betti_0 = []
        train_betti_1 = []

        for eps in np.linspace(0, 1, step_size):
            b_0 = 0
            for k in train_dgm_0:
                if k[0] <= eps and k[1] > eps:
                    b_0 = b_0 + 1
            train_betti_0.append(b_0)

            b_1 = 0
            for l in train_dgm_1:
                if l[0] <= eps and l[1] > eps:
                    b_1 = b_1 + 1
            train_betti_1.append(b_1)

        betti_list.append([dataset] + [graph_id] + graph_label + [
            filtr_time] + train_betti_0 + train_betti_1)  # concatenate betti numbers

        Density = create_traingraph.density()  # obtain density
        Diameter = create_traingraph.diameter()  # obtain diameter
        cluster_coeff = create_traingraph.transitivity_avglocal_undirected()  # obtain transitivity
        laplacian = create_traingraph.laplacian()  # obtain laplacian matrix
        laplace_eigenvalue = np.linalg.eig(laplacian)
        sort_eigenvalue = sorted(np.real(laplace_eigenvalue[0]), reverse=True)
        spectral = sort_eigenvalue[0] - sort_eigenvalue[1]  # obtain spectral gap
        assortativity = create_traingraph.assortativity_degree()  # obtain assortativity
        clique_count = create_traingraph.clique_number()  # obtain clique count
        motifs_count = create_traingraph.motifs_randesu(size=3)  # obtain motif count
        count_components = len(create_traingraph.clusters())  # obtain count components

        density.append(Density)
        diameter.append(Diameter)
        clustering_coeff.append(cluster_coeff)
        spectral_gap.append(spectral)
        assortativity_.append(assortativity)
        cliques.append(clique_count)
        motifs.append(motifs_count)
        components.append(count_components)

    df1 = pd.DataFrame(betti_list)
    df2 = pd.DataFrame(motifs)
    df3 = pd.DataFrame(
        list(zip(density, diameter, clustering_coeff, spectral_gap, assortativity_, cliques, components)))
    feature_data = pd.concat([df1, df2, df3], axis=1, ignore_index=True)

    #    giving column names
    columnnames = {}  # create an empty dict
    count = -4  # initialize count to -1
    for i in feature_data.columns:
        count += 1  # update count by 1
        columnnames[i] = f"eps_{count}"  # index i in dictionary will be named thresh_count

    # rename first and last column in the dictionary
    columnnames.update(
        {(list(columnnames))[0]: 'dataset', (list(columnnames))[1]: 'graphId', (list(columnnames))[2]: 'graphLabel',
         (list(columnnames))[3]: 'filtrTime'})
    feature_data.rename(columns=columnnames, inplace=True)  # give column names to dataframe

    # write dataframe to file
    feature_data.to_csv("save dataframe", index=False)


def main():
    unique_graph_indicator, graph_labels, df_edges, graph_indicators = reading_csv(dataset)
    alpha_filt(unique_graph_indicator, graph_indicators, df_edges, step_size, dataset, graph_labels)


if __name__ == '__main__':
    data_path = "path to data"  # dataset path on computer
    data_list = ('ENZYMES', 'BZR', 'MUTAG', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    for dataset in data_list:
        for step_size in (10, 20, 50, 100):  # we will consider step size 100 for epsilon
            main()
