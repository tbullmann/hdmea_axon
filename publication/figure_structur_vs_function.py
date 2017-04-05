from __future__ import division

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from hana.recording import average_electrode_area

from publication.data import Experiment, FIGURE_CULTURE
from publication.plotting import show_or_savefig, TEMPORARY_PICKELED_NETWORKS, \
    plot_parameter_dependency, format_parameter_plot, compare_networks, analyse_networks, label_subplot, adjust_position, \
    plot_correlation, DataFrame_from_Dicts

logging.basicConfig(level=logging.DEBUG)


def make_figure(figurename, figpath=None):

    # maybe_get_functional_and_structural_networks(FIGURE_ARBORS_FILE, FIGURE_EVENTS_FILE, TEMPORARY_PICKELED_NETWORKS)

    structural_strength, structural_delay, functional_strength, functional_delay \
        = Experiment(FIGURE_CULTURE).networks()

    # Getting and subsetting the data
    data = DataFrame_from_Dicts(functional_delay, functional_strength, structural_delay, structural_strength)
    delayed = data[data.delayed]
    simultaneous = data[data.simultaneous]

    # Map electrode number to area covered by that electrodes
    electrode_area = average_electrode_area(None, mea='hidens')
    structural_strength = {key: electrode_area*value for key, value in structural_strength.items()}

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 13))
    fig.suptitle(figurename + ' Compare structural and functional connectivity', fontsize=14, fontweight='bold')

    # plot network measures
    ax1 = plt.subplot(4,2,1)
    plot_vs_weigth(ax1, structural_strength)
    ax1.set_xlabel(r'$\mathsf{\rho\ [\mu m^2]}$', fontsize=14)
    # ax1.set_title('Structural network')

    ax2 = plt.subplot(4,2,3)
    plot_vs_weigth(ax2, functional_strength)
    ax2.set_xlabel(r'$\mathsf{\zeta}$', fontsize=14)
    # ax2.set_title('Functional network')

    ax3 = plt.subplot(2,2,2)
    structural_thresholds, functional_thresholds, intersection_size, structural_index, jaccard_index, functional_index\
        = compare_networks(structural_strength, functional_strength, scale='log')
    plot_parameter_dependency(ax3, structural_index, structural_thresholds, functional_thresholds,
                              fmt='%1.1f', levels=np.linspace(0, 1, 11))
    format_parameter_plot(ax3)
    # ax3.text(3,120, r'Structural index ${|F \cup S|}/{|S|}$', size=15)
    ax3.set_title(r'$\mathsf{i_S={|F \cup S|}/{|S|}}$', fontsize=14)
    adjust_position(ax3, xshrink=0.02, yshrink=0.02, xshift=0.02, yshift=0.01)

    # plot distribution of strength and delays
    ax4 = plt.subplot(223)
    axScatter1 = plot_correlation(ax4, data, x='structural_strength', y='functional_strength', xscale='log', yscale='log')
    axScatter1.set_xlabel (r'$\mathsf{|A \cup D|\ [\mu m^2}$]', fontsize=14)
    axScatter1.set_ylabel (r'$\mathsf{z_{max}}$', fontsize=14)

    ax5 = plt.subplot(224)
    axScatter2 = plot_correlation(ax5, data,  x='structural_delay', y='functional_delay', xlim=(0, 5), ylim=(0, 5))
    axScatter2.set_xlabel (r'$\mathsf{\tau _{axon}\ [ms]}$', fontsize=14)
    axScatter2.set_ylabel (r'$\mathsf{\tau _{spike}\ [ms]}$', fontsize=14)

    label_subplot(ax1,'A')
    label_subplot(ax2,'B')
    label_subplot(ax3,'C',xoffset=-0.05)
    label_subplot(ax4,'D',xoffset=-0.04)
    label_subplot(ax5,'E',xoffset=-0.04)

    show_or_savefig(figpath, figurename)


def plot_vs_weigth(ax1, dictionary):
    """
    Plot ordinal numbers on the right y axis
    n  = number_of_neurons
    k  = number_of_edges

    Plot real number on the left y axis
    C  = average_clustering
    L  = average_shortest_path_length
    D  = average_degree
    :param ax1:
    :param dictionary: containing pairs of neuron and some weigth (overlap or score)
    """
    w, n, k, C, L, D = analyse_networks(dictionary)

    ax1.plot (w, n, 'k--', label='n')
    ax1.plot (w, k, 'k-', label='k')
    plt.legend(loc=6, frameon=False)
    ax1.set_ylabel('n, k')
    adjust_position(ax1, yshrink=0.01)

    ax2 = ax1.twinx()
    ax2.plot (w, C, 'r-', label='C')
    ax2.plot (w, L, 'g-', label='L')
    ax2.plot (w, D, 'b-', label='D')
    ax2.set_ylabel('C, L, D')
    plt.legend(frameon=False)
    ax1.set_xscale('log')
    ax1.set_xlim((0,max(w)))
    adjust_position(ax2, yshrink=0.01)


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))