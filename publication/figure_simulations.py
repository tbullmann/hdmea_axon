from __future__ import division
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import os

from publication.data import Experiment, FIGURE_CULTURES
from publication.simulation import Simulation
from publication.plotting import show_or_savefig, without_spines_and_ticks
from publication.figure_graphs import plot_scalar_measures, plot_histograms, plot_scalar, flatten

import logging
logging.basicConfig(level=logging.DEBUG)



def make_figure(figurename, figpath=None):

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 8))
    if not figpath:
        fig.suptitle(figurename + '    Compare simulated networks', fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.1)

    plot_scalar_measures(data=Simulation, all_original=False)

    plot_histograms(data=Simulation, all_original=False)

    #sub panel f showing Jaccard index
    # Plot degree distributions for all networks
    df = evaluate_simulations()
    plot_scalar(plt.subplot(278), df, 'Jaccard')
    plt.xlim((1.5,3.5))
    plt.ylim((0, 1))
    plt.ylabel(r'$\mathsf{J}}$', fontsize=16)
    plt.title('f', loc='left', fontsize=18)

    plot_scalar(plt.subplot(279), df, 'TPR')
    plt.xlim((1.5, 3.5))
    plt.ylim((0, 1))
    plt.ylabel(r'$\mathsf{TPR}}$', fontsize=16)
    plt.title('g', loc='left', fontsize=18)

    show_or_savefig(figpath, figurename)


def evaluate_simulations():

    scal = defaultdict(dict)

    for c in FIGURE_CULTURES:

        _, _, functional_strength, functional_delay, \
        synapse_strength, synapse_delay \
            = Experiment(c).networks()

        _, _, simulated_functional_strength, simulated_functional_delay, \
        simulated_synapse_strength, simulated_synapse_delay \
            = Simulation(c).networks()

        scal['function'][c] = indices(functional_delay, simulated_functional_delay)
        scal['synapse'][c] = indices(synapse_delay, simulated_synapse_delay)

    # flatten dictionary with each level stored in key
    df = pd.DataFrame(flatten(scal, ['type', 'culture', 'measure', 'value']))
    df = df.pivot_table(index=['type', 'culture'], columns='measure', values='value')

    return df


def indices (experimental_network, simulated_network):
    experimental_edges = set(experimental_network.keys())
    simulated_edges = set(simulated_network.keys())
    intersection_edges = experimental_edges & simulated_edges
    union_edges = experimental_edges | simulated_edges
    return {'Jaccard': len(intersection_edges) / len(union_edges),
            'TPR': len(intersection_edges) / len(simulated_edges)}



if __name__ == "__main__":
    make_figure(os.path.basename(__file__))
