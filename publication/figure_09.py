import os, pickle

import numpy as np

from hana.matlab import load_neurites
from hana.structure import all_overlaps


# Data preparation

def explore_parameter_space_for_structural_connectivity():
    axon_delay, dendrite_peak = load_neurites ('data/hidens2018at35C_arbors.mat')

    resolution = 3
    alpha = np.float64(range(-5*resolution,5*resolution+1))/resolution
    thresholds_overlap = list((2**alpha)/(2**alpha+1))
    resolution = resolution*2
    thresholds_peak  = list(2**(np.float(exp+1)/resolution)-1 for exp in range(5*resolution+1))

    print 'total', len(thresholds_peak), 'thresholds for peak ', thresholds_peak
    print 'total', len(thresholds_overlap), 'thresholds for overlap = ', thresholds_overlap

    networks = []

    for thr_peak in thresholds_peak:
        for thr_overlap in thresholds_overlap:
            print 'Connections for peak > %1.1f mV and overlap > = %1.2f' % (thr_peak, thr_overlap)
            all_ratio, all_delay = all_overlaps(axon_delay, dendrite_peak, thr_peak=thr_peak, thr_overlap=thr_overlap)
            k = len(all_ratio)
            print 'Network connection k = %d' % k
            networks.append(((thr_peak, thr_overlap),{'overlap ratio': all_ratio, 'delay': all_delay}))

    print 'Finished exploring %d different parameter sets' % len(networks)

    pickle.dump(dict(networks), open('temp/struc_networks_for_thr_peak_thr_overlap_hidens2018.p', 'wb'))

    print 'Saved data'


def analyse_structural_networks():
    network = pickle.load(open('temp/struc_networks_for_thr_peak_thr_overlap_hidens2018.p', 'rb'))
    factors, thresholds = (list(sorted(set(index)) for index in zip(*list(network))))

    k = np.zeros((len(factors), len(thresholds)))
    C = np.zeros((len(factors), len(thresholds)))
    L = np.zeros((len(factors), len(thresholds)))
    D = np.zeros((len(factors), len(thresholds)))
    G = nx.DiGraph()

    for i,thr_peak in enumerate(factors):
        for j,thr_overlap in enumerate(thresholds):
                edges = list(key for key in network[(thr_peak, thr_overlap)]['overlap ratio'])
                G.clear()
                G.add_edges_from(edges)
                giant = max(nx.connected_component_subgraphs(G.to_undirected()), key=len)
                number_of_nodes = nx.number_of_nodes(giant)
                number_of_edges = nx.number_of_edges(giant)
                average_clustering = nx.average_clustering(giant)
                average_shortest_path_length = nx.average_shortest_path_length(giant)
                average_degree = float(number_of_edges)/number_of_nodes
                k[direction][i, j] = number_of_edges
                C[direction][i, j] = average_clustering
                L[direction][i, j] = average_shortest_path_length
                D[direction][i, j] = average_degree
                logging.info('Analyse Graph for thr_peak=%1.3f, thr_overlap=%1.3f' % (thr_peak, thr_overlap))

    pickle.dump((k, C, L, D, factors, thresholds, directions),
                 open('temp/struc_networkparameters_for_thr_peak_thr_overlap_hidens2018.p', 'wb'))


# Final version




if not os.path.isfile('temp/struc_networks_for_thr_peak_thr_overlap_hidens2018.p'): explore_parameter_space_for_structural_connectivity()
if not os.path.isfile('temp/struc_networkparameters_for_thr_peak_thr_overlap_hidens2018.p'): analyse_structural_networks()
