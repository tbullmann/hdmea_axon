from __future__ import division
from collections import defaultdict

from publication.comparison import print_test_result
from publication.data import Experiment, FIGURE_CULTURES
from publication.plotting import show_or_savefig, adjust_position, without_spines_and_ticks

import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
import logging
logging.basicConfig(level=logging.DEBUG)


def make_figure(figurename, figpath=None):

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 13))
    if not figpath:
        fig.suptitle(figurename + '    Compare networks', fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.1)

    plot_scalar_measures()

    plot_histograms()

    ax = plt.subplot(313)
    import matplotlib.image as mpimg
    img = mpimg.imread('data/ComparisonGraphs.png')
    plt.imshow(img)
    plt.axis('off')
    plt.title('j', loc='left', fontsize=18)
    ax.set_anchor('W')
    adjust_position(ax, xshift=0.02,yshift=-0.03)

    show_or_savefig(figpath, figurename)


def plot_histograms(data=Experiment, all_original=True):
    hist = defaultdict(dict)
    for c in FIGURE_CULTURES:
        structural_strength, structural_delay, functional_strength, functional_delay, synapse_strength, synapse_delay \
            = data(c).networks()
        hist['structure'][c] = histograms(structural_delay)
        hist['function'][c] = histograms(functional_delay)
        hist['synapse'][c] = histograms(synapse_delay)

    # Plot degree distributions for all networks
    if all_original:
        ax337 = plt.subplot(345)
        plot_degree_distributions(hist, 'structure', 'b')
        plt.title('g', loc='left', fontsize=18)
        without_spines_and_ticks(ax337)
    ax338 = plt.subplot(346)
    plot_degree_distributions(hist, 'function', 'r')
    if all_original:
        plt.title('h', loc='left', fontsize=18)
    else:
        plt.title('i', loc='left', fontsize=18)
        adjust_position(ax338, xshift=0.02, yshrink=0.01)
    without_spines_and_ticks(ax338)

    ax339 = plt.subplot(347)
    plot_degree_distributions(hist, 'synapse', 'g')
    if all_original:
        plt.title('i', loc='left', fontsize=18)
    else:
        plt.title('j', loc='left', fontsize=18)
        adjust_position(ax339, xshift=0.02, yshrink=0.01)
    without_spines_and_ticks(ax339)

    ax248 = plt.subplot(348)
    if all_original:
        plt.scatter(None, None, 20, color='blue', marker='s', label='structural')
        plt.scatter(None, None, 20, color='red', marker='s', label='functional')
        plt.scatter(None, None, 20, color='green', marker='s', label='effective')
    else:
        plt.scatter(None, None, 20, color='blue', marker='s', label='original\nstructural')
        plt.scatter(None, None, 20, color='red', marker='s', label='simulated\nfunctional')
        plt.scatter(None, None, 20, color='green', marker='s', label='simulated\neffective')

    plt.axis('off')
    plt.legend(loc='upper center', scatterpoints=1, markerscale=3., frameon=False)


def plot_scalar_measures(data=Experiment, all_original=True):
    # --- Scalar measures
    scal = defaultdict(dict)
    for c in FIGURE_CULTURES:
        structural_strength, structural_delay, functional_strength, functional_delay, synapse_strength, synapse_delay \
            = data(c).networks()

        scal['structure'][c] = scalar_measures(structural_delay)
        scal['function'][c] = scalar_measures(functional_delay)
        scal['synapse'][c] = scalar_measures(synapse_delay)

    print (scal)

    # flatten dictionary with each level stored in key
    df = pd.DataFrame(flatten(scal, ['type', 'culture', 'measure', 'value']))

    print (df)

    df = df.pivot_table(index=['type', 'culture'], columns='measure', values='value')
    plot_scalar(plt.subplot(361), df, 'mean_p')
    plt.ylim((0, 1))
    plt.ylabel(r'average connection density  $\mathsf{p}}$', fontsize=12)
    plt.title('a', loc='left', fontsize=18)
    plot_scalar(plt.subplot(362), df, 'rud')
    plt.ylim((0, 1))
    plt.ylabel(r'unidirectionality  $\mathsf{r_{uni}}$', fontsize=12)
    plt.title('b', loc='left', fontsize=18)
    plot_scalar(plt.subplot(363), df, 'r')
    plt.ylabel(r'assortativity  $\mathsf{R}$', fontsize=12)
    plt.ylim((-1, 1))
    plt.title('c', loc='left', fontsize=18)
    plot_scalar(plt.subplot(364), df, 'c')
    # print ('structural:')
    # print (df['c']['structure'])
    # print ('functional:')
    # print (df['c']['function'])
    # print ('effective:')
    # print (df['c']['synapse'])
    plt.ylabel(r'average clustering coefficient  $\mathsf{C}$', fontsize=12)
    plt.ylim((0, 1))
    plt.title('d', loc='left', fontsize=18)
    plot_scalar(plt.subplot(365), df, 'l')
    plt.ylim((0, 3))
    plt.ylabel(r'characteristic path length  $\mathsf{L}$', fontsize=12)
    plt.title('e', loc='left', fontsize=18)
    plot_scalar(plt.subplot(366), df, 'SWS')
    plt.ylim((0, 2))
    plt.ylabel(r'small world-ness  $\mathsf{S_{ws}}$', fontsize=12)
    plt.title('f', loc='left', fontsize=18)


def plot_scalar(ax, df, scalar_name):
    xorder = ['structure', 'function', 'synapse']
    scalar = df[scalar_name].unstack(level=-1).reindex(xorder)
    if scalar_name == 'r':
        print_test_result('assortativity of structural connectivity equal 0', df.loc['structure'], 0)
        print_test_result('assortativity of functional connectivity equal 0', df.loc['function'], 0)
        print_test_result('assortativity of synaptic connectivity equal 0', df.loc['synapse'], 0)
    if scalar_name == 'r':
        print_test_result('small-world-ness of structural connectivity equal 1', df.loc['structure'], 1)
        print_test_result('small-world-ness of functional connectivity equal 1', df.loc['function'], 1)
        print_test_result('small-world-ness of synaptic connectivity equal 1', df.loc['synapse'], 1)

    logging.info('scalar %s:' % scalar_name)
    logging.info(scalar)
    bplot = ax.boxplot(np.array(scalar).T,
                       boxprops={'color': "black", "linestyle": "-"}, patch_artist=True, widths=0.7)
    prettify_boxplot(ax, bplot)


def plot_degree_distributions(hist, network_type, color):
    for culture in FIGURE_CULTURES:
        plt.plot(hist[network_type][culture]['ps'], hist[network_type][culture]['p_in'], color=color)
        plt.plot(hist[network_type][culture]['ps'], hist[network_type][culture]['p_out'], color=color, linestyle=':')
    plt.plot(0, 0, color=color, label='in')
    plt.plot(0, 0, color=color, linestyle=':', label='out')
    plt.legend(loc = 'upper center', frameon=False)
    plt.xlabel(r'$\mathsf{p_{in}, p_{out}}$', fontsize=16)
    plt.ylim((0,6))
    plt.ylabel('density')
    adjust_position(plt.gca(), xshift=0.01, xshrink=0.01)



def flatten(data, group_names):
    # https://codereview.stackexchange.com/questions/129135/flatten-a-nested-dict-structure-in-python
    # answered May 25 '16 at 12:54 by Mathias Ettinger
    try:
        group, group_names = group_names[0], group_names[1:]
    except IndexError:
        # No more key to extract, we just reached the most nested dictionnary
        yield data.copy()  # Build a new dict so we don't modify data in place
        return  # Nothing more to do, it is already considered flattened

    try:
        for key, value in data.items():
            # value can contain nested dictionaries
            # so flatten it and iterate over the result
            for flattened in flatten(value, group_names):
                flattened.update({group: key})
                yield flattened
    except AttributeError:
        yield {group: data}


def scalar_measures(edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    G.remove_edges_from((G.selfloop_edges()))
    undirected_G = G.to_undirected()
    Gc = max(nx.connected_component_subgraphs(undirected_G), key=len)

    m = G.size()
    n = G.order()
    mean_k = m / n

    # density
    mean_p = mean_k / (n - 1)

    # assortativity
    r = nx.degree_assortativity_coefficient(G)

    # average clustering coefficient
    c = nx.average_clustering(undirected_G)

    # average shortest path
    l = nx.average_shortest_path_length(Gc)

    # edge dispersion --- NOT USED
    in_degrees = list(dict(G.in_degree).values())
    CV_k_in = np.std(in_degrees)/np.mean(in_degrees)
    out_degrees = list(dict(G.out_degree).values())
    CV_k_out = np.std(out_degrees)/np.mean(out_degrees)

    # small worldness
    SWS = small_worldness(Gc)

    # ratio of unidirectional connections
    rud = ratio_unidirectional(edges)

    return {'mean_p': mean_p,
               'r': r,
               'c': c,
               'l': l,
               'CV_k_in': CV_k_in,
               'CV_k_out': CV_k_out,
               'SWS': SWS,
               'rud': rud}


def histograms(edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    G.remove_edges_from((G.selfloop_edges()))
    undirected_G = G.to_undirected()

    n = G.order()

    ps, p_in = estimate_wiring_p(list(dict(G.in_degree).values()), n)
    ps, p_out = estimate_wiring_p(list(dict(G.out_degree).values()), n)

    rc = nx.rich_club_coefficient(undirected_G, normalized=False)

    return {'p_in': p_in, 'p_out': p_out, 'ps': ps,
                  'rc': rc}


def small_worldness(G, num=100):
    """
    Calculates the small-world-ness according to Humphries & Gurney, 2008.
    :param G: undirected, connected graph
    :param num: number of surrogate Erdoes-Renyi type random networks
    :return: SWS: small-world-ness
    """
    # n nodes and m edges
    n = G.order()
    m = G.size()

    # equivalent Erdoes-Renyi (E-R) random graph with the same m and n
    Crand = list()
    Lrand = list()
    for _ in range(num):
        Grand = nx.gnm_random_graph(n, m)
        Crand.append(nx.average_clustering(Grand))
        Lrand.append(nx.average_shortest_path_length(G))
    Crand = np.median(Crand)
    Lrand = np.median(Lrand)

    # average clustering coefficients
    C = nx.average_clustering(G)

    # average shortest paths
    L = nx.average_shortest_path_length(G)

    SWS = (L/Lrand)/(C/Crand)

    return SWS


def estimate_wiring_p(degrees, n):
    in_p = np.array(degrees) / (n - 1)  # probability
    print(in_p)
    gkde = stats.gaussian_kde(in_p)
    ps = np.linspace(0, 1, 100)
    h = gkde.evaluate(ps)
    return ps, h


def prettify_boxplot(ax, bplot):
    colors = ['blue', 'red', 'green']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_color(color)
    plt.setp(bplot['medians'], color='black')
    plt.setp(bplot['whiskers'], color='black')
    plt.xlabel('network type')
    # ax.set_xticks([1, 2, 3])
    # ax.set_xticklabels(['structure','function', 'synapse'])
    adjust_position(ax,xshrink=0.02, yshrink=0.01)
    without_spines_and_ticks(ax)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are off


def test_connectivity (data=Experiment, all_original=True):
    # --- Scalar measures
    scal = defaultdict(dict)
    for c in FIGURE_CULTURES:
        structural_strength, structural_delay, functional_strength, functional_delay, synapse_strength, synapse_delay \
            = data(c).networks()

        count_FFT(structural_delay)
        count_FFT(functional_delay)
        count_FFT(synapse_delay)


def count_FFT(connectivity):
    """
    Counting the number of feed-forward triangles, that are chain & shortcut motives like A-->B-->C & A-->C.

    Note: According to U.Alon's book ("Introduction to Systems Biology"), the mean number of times that a
    subgraph G with n edges occurs in a random network is given by the following formula:

        <N_G> = 1/alpha * N^n * p^n

    alpha: is a number that includes combinatorial factors related to the structure and symmetry of each subgraph,
    equal to 1 for feed-forward loop (FFL) and equal to 3 for feed-back loop (FBL)
    N^n is the number of ways of choosing a set of n nodes out of N. Because there are N ways of choosing the first
    one, times N-1 ways of choosing the second one, and so on..(so N^n approximation is true for large networks)
    p^g is the probability to get the g edges in the appropriate places (each with probability p)

    With k = pN, the number of edges it transformes to

        <N_G> = 1/alpha * k^n    or better <N_G> + 1/alpha * k * (k-1) * ... * (k-n)

    :param connectivity:
    :return:
    """
    observed_triangles = 0
    errors = list()
    for (pre, post) in connectivity:
        for (pre2, intermediate) in connectivity:
            if pre == pre2 and ((intermediate, post) in connectivity):
                observed_triangles += 1
                errors.append(connectivity[(pre, intermediate)] + connectivity[(intermediate, post)]
                              - connectivity[(pre, post)])

    # score is the log10 of observed/expected triangles
    k = len(connectivity)
    expected_triangles = k * (k-1) * (k-2)
    score = np.log10(observed_triangles / expected_triangles)

    print (observed_triangles, score, np.mean(errors), np.median(errors))


def ratio_unidirectional(connectivity):
    """
    Neurons are connected unidirectional instead of bidirectional. In other words, in most cases there is a pre- and
    a post-synaptic neuron and not pairs that are pre- (and post-synaptic) to each other. One can argue, that spurios
    connections are most likely bidirectional.
    :param connectivity:
    :return: ratio of unidirectional connections in all connections
    """

    number_bidirectional = 0
    number_unidirectional = 0

    for (pre, post) in connectivity:
        if (post, pre) in connectivity:
            number_bidirectional += 1
        else:
            number_unidirectional += 1

    return (number_unidirectional/(number_bidirectional+number_unidirectional))


if __name__ == "__main__":
    # make_figure(os.path.basename(__file__))
    # test_connectivity()
    a = {
        'foo': {
            'cat': {'name': 'Hodor', 'age': 7},
            'dog': {'name': 'Mordor', 'age': 5},
        },
        'bar': {
            'rat': {'name': 'Izidor', 'age': 3},
        },
        'baz': 'woops',
    }
    print(a)
    print(list(flatten(a, ['foobar', 'animal'])))
