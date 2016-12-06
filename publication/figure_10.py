from hana.function import timeseries_to_surrogates, all_timelag_standardscore, all_peaks
from hana.recording import load_timeseries
from publication.plotting import plot_parameter_dependency, FIGURE_EVENTS_FILE

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import os, pickle, logging
logging.basicConfig(level=logging.DEBUG)


# Data preparation

def prepare_timeseries_for_figures():
    timeseries =load_timeseries(FIGURE_EVENTS_FILE)
    pickle.dump((timeseries), open('temp/timeseries.p', 'wb'))


def explore_parameter_space_for_functional_connectivity():
    timeseries = pickle.load(open('temp/timeseries.p', 'rb'))

    n=20 # number of surrogate data (if too low, std is not robust)
    resolution = 4
    resolution = 1 # for testing
    factors = list(2**(np.float(exp)/resolution) for exp in range(6*resolution+1))
    logging.info('Factors: ')
    logging.info(factors)
    resolution = 2 # for testing
    thresholds = list(2**(np.float(exp)/resolution) for exp in range(6*resolution+1))
    directions = ('forward','reverse')

    network_size = []
    networks = []

    for factor in factors:
        logging.info('Surrogate data (n = %d const.) using factor = %f' % (n, factor))
        timeseries_surrogates = timeseries_to_surrogates(timeseries, n=n, factor=factor )
        logging.info('Compute standard score for histograms')
        timelags, std_score_dict, timeseries_hist_dict = all_timelag_standardscore(timeseries, timeseries_surrogates)
        for thr in thresholds:
            for direction in directions:
                logging.info('Peaks for threshold = %f and direction = %s' % (thr, direction))
                peak_score, peak_timelag, _, _ = all_peaks(timelags, std_score_dict, thr=thr, direction=direction)
                k = len(peak_score)
                logging.info('Network connection k = %d' % k)
                network_size.append(((factor, thr, direction), k))
                networks.append(((factor, thr, direction),{'peak_score': peak_score, 'peak_timelag': peak_timelag}))

    logging.info('Finshed exploring %d different parameter sets' % len(network_size))

    pickle.dump(dict(network_size), open('temp/functional_networks_size.p', 'wb'))
    pickle.dump(dict(networks), open('temp/functional_networks.p', 'wb'))

    logging.info('Saved data')


def analyse_functional_networks():
    network = pickle.load(open('temp/functional_networks.p', 'rb'))
    factors, thresholds, directions = (list(sorted(set(index)) for index in zip(*list(network))))

    k = {'forward':np.zeros((len(factors), len(thresholds))),'reverse':np.zeros((len(factors), len(thresholds)))}
    C = {'forward':np.zeros((len(factors), len(thresholds))),'reverse':np.zeros((len(factors), len(thresholds)))}
    L = {'forward':np.zeros((len(factors), len(thresholds))),'reverse':np.zeros((len(factors), len(thresholds)))}
    D = {'forward':np.zeros((len(factors), len(thresholds))),'reverse':np.zeros((len(factors), len(thresholds)))}
    G = nx.DiGraph()

    for i,factor in enumerate(factors):
        for j,thr in enumerate(thresholds):
            for direction in directions:
                edges = list(key for key in network[(factor, thr, direction)]['peak_score'])
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
                logging.info('Analyse Graph for factor=%1.3f, threshold=%1.3f, direction: %s' % (factor, thr, direction))

    pickle.dump((k, C, L, D, factors, thresholds, directions), open('temp/functional_networks_parameters.p', 'wb'))

# Early version

def load_number_of_edges_functional_networksize():
    network_size = pickle.load(open('temp/functional_networks_size.p', 'rb'))
    factors, thresholds, directions = (list(sorted(set(index)) for index in zip(*list(network_size))))
    k = {'forward':np.zeros((len(factors), len(thresholds))),'reverse':np.zeros((len(factors), len(thresholds)))}
    for i,factor in enumerate(factors):
        for j,thr in enumerate(thresholds):
            k['forward'][i,j] = network_size[(factor, thr ,'forward')]
            k['reverse'][i,j] = network_size[(factor, thr ,'reverse')]
    return k, factors, thresholds, directions


def figure_10_only_k():
    """
    Plots the number of edges functional network, which were saved during parameter space exploration.
    This was used mainly for testing.
    """
    plt.figure()
    ax = plt.subplot(111)
    k, factors, thresholds, directions = load_number_of_edges_functional_networksize()
    plot_parameter_dependency(ax, k, factors, thresholds, directions, levels=(5, 10, 20, 50, 100, 250, 500))
    ax.set_title('number of edges, k')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('threshold $\zeta$')
    ax.set_ylabel('randomization factor $\sigma$')
    plt.show()


# Final version

def scale_and_label(ax, line_x_coordinates, line_y_coordinates):
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('randomization factor, $\sigma$')
    ax.set_ylabel('threshold for score, $\zeta$')
    ax.plot(line_x_coordinates, line_y_coordinates, 'k:')


def figure10():
    plt.figure('Figure 10', figsize=(12,12))

    k,C,L,D,factors, thresholds, directions = pickle.load(open('temp/functional_networks_parameters.p', 'rb'))

    line_x_coordiantes = (min(factors), max(factors))
    line_y_coordinates = (max(thresholds), min(thresholds))

    logging.info('Plotting results for %d paramter sets' % len(k['forward'])**2)

    ax = plt.subplot(221)
    plot_parameter_dependency(ax, k, factors, thresholds, directions, levels=(5, 10, 20, 50, 100, 250, 500))
    ax.set_title('number of edges, k')
    scale_and_label(ax, line_x_coordiantes, line_y_coordinates)

    ax = plt.subplot(222)
    plot_parameter_dependency(ax, C, factors, thresholds, directions, fmt='%1.1f')
    ax.set_title('average clustering coefficient, C')
    scale_and_label(ax, line_x_coordiantes, line_y_coordinates)

    ax = plt.subplot(223)
    plot_parameter_dependency(ax, L, factors, thresholds, directions, fmt='%1.1f')
    ax.set_title('average maximum pathlength, L')
    scale_and_label(ax, line_x_coordiantes, line_y_coordinates)

    ax = plt.subplot(224)
    plot_parameter_dependency(ax, D, factors, thresholds, directions, fmt='%1.1f')
    ax.set_title('average node degree, D')
    scale_and_label(ax, line_x_coordiantes, line_y_coordinates)

    plt.show()


if not os.path.isfile('temp/timeseries.p'): prepare_timeseries_for_figures()
if not os.path.isfile('temp/functional_networks.p'): explore_parameter_space_for_functional_connectivity()
if not os.path.isfile('temp/functional_networks_parameters.p'):  analyse_functional_networks()

# figure_10_only_k()

figure10()
