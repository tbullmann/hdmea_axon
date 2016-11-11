import logging
import pickle

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from hana.function import timeseries_to_surrogates, all_timelag_standardscore, all_peaks
from hana.matlab import events_to_timeseries

logging.basicConfig(level=logging.DEBUG)

def prepare_timeseries_for_figures():
    events = load_events('data/hidens2018at35C_events.mat')
    timeseries = events_to_timeseries(events)
    pickle.dump((timeseries), open('temp/timeseries_hidens2018.p', 'wb'))


def figure11_prepare_data():
    timeseries = pickle.load(open('temp/timeseries_hidens2018.p', 'rb'))

    n=20 # number of surrogate data (if too low, std is not robust)
    resolution = 4
    factors = list(2**(np.float(exp)/resolution) for exp in range(6*resolution+1))
    print factors
    thresholds = list(2**(np.float(exp)/resolution) for exp in range(6*resolution+1))
    directions = ('forward','reverse')

    network_size = []
    networks = []

    for factor in factors:
        print 'Surrogate data (n = %d const.) using factor = %f' % (n, factor)
        timeseries_surrogates = timeseries_to_surrogates(timeseries, n=n, factor=factor )
        print 'Compute standard score for histograms'
        timelags, std_score_dict, timeseries_hist_dict = all_timelag_standardscore(timeseries, timeseries_surrogates)
        for thr in thresholds:
            for direction in directions:
                print 'Peaks for threshold = %f and direction = %s' % (thr, direction)
                peak_score, peak_timelag, _, _ = all_peaks(timelags, std_score_dict, thr=thr, direction=direction)
                k = len(peak_score)
                print 'Network connection k = %d' % k
                network_size.append(((factor, thr, direction), k))
                networks.append(((factor, thr, direction),{'peak_score': peak_score, 'peak_timelag': peak_timelag}))

    print 'Finshed exploring %d different parameter sets' % len(network_size)

    pickle.dump(dict(network_size), open('temp/func_networksize_for_factor_thr_direction_hidens2018.p', 'wb'))
    pickle.dump(dict(networks), open('temp/func_networks_for_factor_thr_direction_hidens2018.p', 'wb'))

    print 'Saved data'


def load_number_of_edges_functional_networksize():
    network_size = pickle.load(open('temp/func_networksize_for_factor_thr_direction_hidens2018.p', 'rb'))
    factors, thresholds, directions = (list(sorted(set(index)) for index in zip(*list(network_size))))
    k = {'forward':np.zeros((len(factors), len(thresholds))),'reverse':np.zeros((len(factors), len(thresholds)))}
    for i,factor in enumerate(factors):
        for j,thr in enumerate(thresholds):
            k['forward'][i,j] = network_size[(factor, thr ,'forward')]
            k['reverse'][i,j] = network_size[(factor, thr ,'reverse')]
    return k, factors, thresholds, directions


def plot_parameter_dependency(ax, Z, x, y, w=None, levels=None, fmt='%d'):
    """Plotting parameter dependency"""
    from scipy.ndimage.filters import gaussian_filter
    import matplotlib.patches as mpatches
    X, Y = np.meshgrid(x, y)
    if w is None:  # standard black and white blot
        CS1 = ax.contour(X, Y, gaussian_filter(Z, 1), levels=levels, colors='k')
        ax.clabel(CS1, fmt=fmt, inline=1, fontsize=10, inline_spacing=-5)
    else:  # first entry entry in dictionary in black, second in red and dashed lines
        CS1 = ax.contour(X, Y, gaussian_filter(Z[w[0]], 1), levels=levels, colors='k')
        ax.clabel(CS1, fmt=fmt, inline=1, fontsize=10, inline_spacing=-5)
        CS2 = ax.contour(X, Y, gaussian_filter(Z[w[1]], 1), levels=levels, colors='r',
                         linestyles='dashed')
        ax.clabel(CS2, fmt=fmt, inline=1, fontsize=10, inline_spacing=-5)
        # Zinf = np.float64(np.isfinite(Z[w[0]]))
        # CS1 = ax.contourf(X, Y, gaussian_filter(Zinf, 1), levels=(0,1), colors='y')
        black_patch = mpatches.Patch(color='black', label=w[0])
        red_patch = mpatches.Patch(color='red', label=w[1])
        plt.legend(loc=2, handles=[black_patch, red_patch])


def plot_number_of_edges_functional_network(ax):
    k, factors, thresholds, directions = load_number_of_edges_functional_networksize()
    plot_parameter_dependency(ax, k, factors, thresholds, directions,levels=(5, 10, 20, 50, 100, 250, 500))


def figure11():
    plt.figure()

    k,C,L,D,factors, thresholds, directions = pickle.load(open('temp/func_networkparameters_for_factor_thr_direction_hidens2018.p', 'rb'))

    print "Plotting results for %d paramter sets" % len(k['forward'])**2

    ax = plt.subplot(221)
    plot_parameter_dependency(ax, k, factors, thresholds, directions,levels=(5, 10, 20, 50, 100, 250, 500))
    ax.set_title('number of edges, k')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('threshold $\zeta$')
    ax.set_ylabel('randomization factor $\sigma$')

    ax = plt.subplot(222)
    plot_parameter_dependency(ax, C, factors, thresholds, directions, fmt='%1.1f')
    ax.set_title('average clustering coefficient, C')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('threshold $\zeta$')
    ax.set_ylabel('randomization factor $\sigma$')

    ax = plt.subplot(223)
    plot_parameter_dependency(ax, L, factors, thresholds, directions, fmt='%1.1f')
    ax.set_title('average maximum pathlength, L')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('threshold $\zeta$')
    ax.set_ylabel('randomization factor $\sigma$')

    ax = plt.subplot(224)
    plot_parameter_dependency(ax, D, factors, thresholds, directions, fmt='%1.1f')
    ax.set_title('average node degree, D')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('threshold $\zeta$')
    ax.set_ylabel('randomization factor $\sigma$')

    plt.show()

def extract_functional_networks():
    network = pickle.load(open('temp/func_networks_for_factor_thr_direction_hidens2018.p', 'rb'))
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
    pickle.dump((k, C, L, D, factors, thresholds, directions),
                 open('temp/func_networkparameters_for_factor_thr_direction_hidens2018.p', 'wb'))




# prepare_timeseries_for_figures()
# figure11_prepare_data()
# extract_functional_networks()
figure11()
