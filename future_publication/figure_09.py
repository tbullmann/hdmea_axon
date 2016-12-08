from hana.segmentation import load_neurites
from hana.structure import all_overlaps
from publication.plotting import plot_parameter_dependency, FIGURE_ARBORS_FILE

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import os, pickle, logging
logging.basicConfig(level=logging.DEBUG)

# Data preparation

def explore_parameter_space_for_structural_connectivity():
    axon_delay, dendrite_peak = load_neurites (FIGURE_ARBORS_FILE)

    resolution = 3
    alpha = np.float64(range(-5*resolution,5*resolution+1))/resolution
    thresholds_ratio = list((2**alpha)/(2**alpha+1))
    resolution = resolution*2
    thresholds_peak  = list(2**(np.float(exp+1)/resolution)-1 for exp in range(5*resolution+1))

    logging.info('Total', len(thresholds_peak), 'thresholds for peak ', thresholds_peak)
    logging.info('Total', len(thresholds_ratio), 'thresholds for overlap = ', thresholds_ratio)

    networks = []

    for thr_peak in thresholds_peak:
        for thr_ratio in thresholds_ratio:
            logging.info('Connections for peak > %1.1f mV and overlap > = %1.2f' % (thr_peak, thr_ratio))
            all_overlap, all_ratio, all_delay = all_overlaps(axon_delay, dendrite_peak, thr_peak=thr_peak,
                                                             thr_ratio=thr_ratio)
            k = len(all_ratio)
            logging.info('Network connection k = %d' % k)
            networks.append(((thr_peak, thr_ratio), {'overlap ratio': all_ratio, 'delay': all_delay}))

    logging.info('Finished exploring %d different parameter sets' % len(networks))

    pickle.dump(dict(networks), open('temp/structural_networks.p', 'wb'))

    logging.info('Saved data')


def analyse_structural_networks():
    network = pickle.load(open('temp/structural_networks.p', 'rb'))
    thresholds_peak, thresholds_ratio = (list(sorted(set(index)) for index in zip(*list(network))))

    k = np.zeros((len(thresholds_peak), len(thresholds_ratio)))
    C = np.zeros((len(thresholds_peak), len(thresholds_ratio)))
    L = np.zeros((len(thresholds_peak), len(thresholds_ratio)))
    D = np.zeros((len(thresholds_peak), len(thresholds_ratio)))
    G = nx.DiGraph()

    for i,thr_peak in enumerate(thresholds_peak):
        for j,thr_ratio in enumerate(thresholds_ratio):
                logging.info('Analyse Graph for thr_peak=%1.3f, thr_overlap=%1.3f' % (thr_peak, thr_ratio))
                edges = list(key for key in network[(thr_peak, thr_ratio)]['overlap ratio'])
                G.clear()
                G.add_edges_from(edges)
                logging.info('k=%d', len(edges))
                if G:
                    giant = max(nx.connected_component_subgraphs(G.to_undirected()), key=len)
                    number_of_nodes = nx.number_of_nodes(giant)
                    number_of_edges = nx.number_of_edges(giant)
                    average_clustering = nx.average_clustering(giant)
                    average_shortest_path_length = nx.average_shortest_path_length(giant)
                    average_degree = float(number_of_edges)/number_of_nodes
                    k[i, j] = number_of_edges
                    C[i, j] = average_clustering
                    L[i, j] = average_shortest_path_length
                    D[i, j] = average_degree

    pickle.dump((k, C, L, D, thresholds_peak, thresholds_ratio), open('temp/structural_networks_parameters.p', 'wb'))


# Final version

def scale_and_label(ax, line_x_coordinates, line_y_coordinates):
    ax.set_xlabel('threshold for positive peak, $\\rho$ [mV]')
    ax.set_ylabel('threshold for overlap, $\phi$')
    ax.plot(line_x_coordinates, line_y_coordinates, 'k:')


def figure09():
    plt.figure('Figure 9', figsize=(12,12))

    k, C, L, D, thr_peak, thr_ratio = pickle.load(open('temp/structural_networks_parameters.p', 'rb'))

    line_x_coordiantes = (thr_peak[14], thr_peak[14])  # see figure_11.compare_structural_and_functional_networks
    line_y_coordinates = (min(thr_ratio), max(thr_ratio))

    logging.info('Plotting results for %d paramter sets' % len(k)**2)

    ax = plt.subplot(221)
    plot_parameter_dependency(ax, k, thr_peak, thr_ratio, levels=(5, 10, 20, 50, 100, 250, 500))
    ax.set_title('number of edges, k')
    scale_and_label(ax, line_x_coordiantes, line_y_coordinates)

    ax = plt.subplot(222)
    plot_parameter_dependency(ax, C, thr_peak, thr_ratio, fmt='%1.1f')
    ax.set_title('average clustering coefficient, C')
    scale_and_label(ax, line_x_coordiantes, line_y_coordinates)

    ax = plt.subplot(223)
    plot_parameter_dependency(ax, L, thr_peak, thr_ratio, fmt='%1.1f')
    ax.set_title('average maximum pathlength, L')
    scale_and_label(ax, line_x_coordiantes, line_y_coordinates)

    ax = plt.subplot(224)
    plot_parameter_dependency(ax, D, thr_peak, thr_ratio, fmt='%1.1f')
    ax.set_title('average node degree, D')
    scale_and_label(ax, line_x_coordiantes, line_y_coordinates)

    plt.show()


if not os.path.isfile('temp/structural_networks.p'): explore_parameter_space_for_structural_connectivity()
if not os.path.isfile('temp/structural_networks_parameters.p'): analyse_structural_networks()

figure09()
