from __future__ import division
import logging
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from hana.misc import unique_neurons
from hana.function import timeseries_to_surrogates, all_timelag_standardscore, all_peaks
from hana.recording import load_timeseries, average_electrode_area
from hana.segmentation import load_neurites, load_compartments, load_positions, neuron_position_from_trigger_electrode
from hana.plotting import plot_network, plot_neuron_points, mea_axes
from hana.structure import all_overlaps
from publication.plotting import show_or_savefig, FIGURE_ARBORS_FILE, FIGURE_EVENTS_FILE, TEMPORARY_PICKELED_NETWORKS, \
    plot_parameter_dependency, format_parameter_plot, compare_networks, analyse_networks, label_subplot, adjust_position, \
    correlate_two_dicts, kernel_density, axes_to_3_axes
from publication.figure_structur_vs_function import maybe_get_functional_and_structural_networks
from publication.figure_synapse import plot_synapse_delays

logging.basicConfig(level=logging.DEBUG)

def make_figure(figurename, figpath=None):

    maybe_get_functional_and_structural_networks(FIGURE_ARBORS_FILE, FIGURE_EVENTS_FILE, TEMPORARY_PICKELED_NETWORKS)

    structural_strengths, structural_delays, functional_strengths, functional_delays \
        = pickle.load( open(TEMPORARY_PICKELED_NETWORKS, 'rb'))

    # Map electrode number to area covered by that electrodes
    electrode_area = average_electrode_area(None, mea='hidens')
    structural_strengths = {key: electrode_area*value for key, value in structural_strengths.items()}

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 13))
    fig.suptitle(figurename + ' Compare structural and functional connectivity', fontsize=14, fontweight='bold')

    # plot network measures
    ax1 = plt.subplot(4,2,1)
    plot_vs_weigth(ax1, structural_strengths)
    ax1.set_xlabel(r'$\mathsf{\rho\ [\mu m^2]}$', fontsize=14)
    # ax1.set_title('Structural network')

    ax2 = plt.subplot(4,2,3)
    plot_vs_weigth(ax2, functional_strengths)
    ax2.set_xlabel(r'$\mathsf{\zeta}$', fontsize=14)
    # ax2.set_title('Functional network')

    ax3 = plt.subplot(2,2,2)
    adjust_position(ax3, xshift=0.04, yshift=-0.01)
    # plot distribution of strength
    axScatter1 = plot_correlation(ax3, structural_strengths, functional_strengths, xscale='log', yscale='log')
    axScatter1.set_xlabel (r'$\mathsf{|A \cup D|\ [\mu m^2}$]', fontsize=14)
    axScatter1.set_ylabel (r'$\mathsf{z_{max}}$', fontsize=14)



    ax4 = plt.subplot(223)
    delayed_pairs, simultaneous_pairs, synpase_delays = plot_synapse_delays(ax4, structural_delays, functional_delays,
                                                                            functional_strengths, ylim=(-2, 7))

    ax5 = plt.subplot(224)
    trigger, _, _, _ = load_compartments(FIGURE_ARBORS_FILE)
    pos = load_positions(mea='hidens')
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)
    plot_network(ax5, simultaneous_pairs, neuron_pos, color='red')
    plot_network(ax5, delayed_pairs, neuron_pos, color='green')
    plot_neuron_points(ax5, unique_neurons(structural_delays), neuron_pos)
    # plot_neuron_id(ax2, trigger, neuron_pos)
    # Legend by proxy
    ax5.hlines(0, 0, 0, linestyle='-', color='red', label='<1ms')
    ax5.hlines(0, 0, 0, linestyle='-', color='green', label='>1ms')
    ax5.text(200, 200, r'$\mathsf{\rho=300\mu m^2}$', fontsize=14)
    plt.legend(frameon=False)
    mea_axes(ax5)

    # Label subplots


    label_subplot(ax1,'A')
    label_subplot(ax2,'B')
    label_subplot(ax3,'C',xoffset=-0.05)
    label_subplot(ax4, 'D', xoffset=-0.04, yoffset=-0.02)
    label_subplot(ax5, 'E', xoffset=-0.02, yoffset=-0.02)

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


def plot_correlation(ax, xdict, ydict, best_keys=None, xlim = None, ylim = None, dofit=False, xscale='linear', yscale='linear', scaling = 'count'):
    """Plot correlation as scatter plot and marginals"""
    # getting the data
    x_all = xdict.values()
    y_all = ydict.values()
    x_corr, y_corr = correlate_two_dicts(xdict, ydict)
    if best_keys: x_best, y_best = correlate_two_dicts(xdict, ydict, best_keys)
    # new axes
    rect_histx, rect_histy, rect_scatter = axes_to_3_axes(ax)
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    # the scatter plot:
    axScatter.scatter(x_corr, y_corr, color='black')
    if best_keys: axScatter.scatter(x_best, y_best, color='red')
    # plt.sca(axScatter)
    # plt.legend(frameon=False, loc=2)
    # the marginals
    kernel_density(axHistx, x_all, scaling=scaling, style='k:')
    kernel_density(axHisty, y_all, scaling=scaling, style='k:', orientation='horizontal')
    kernel_density(axHistx, x_corr, scaling=scaling, style='k--')
    kernel_density(axHisty, y_corr, scaling=scaling, style='k--', orientation='horizontal')
    if best_keys:
        kernel_density(axHistx, x_best, scaling=scaling, style='r-')
        kernel_density(axHisty, y_best, scaling=scaling, style='r-', orientation='horizontal')
    # joint legend by proxies
    plt.sca(ax)
    plt.vlines(0, 0, 0, colors='black', linestyles=':', label='all')
    plt.vlines(0, 0, 0, colors='black', linestyles='--', label='both')
    if best_keys: plt.vlines(0, 0, 0, colors='red', linestyles='-', label='best')
    plt.vlines(0, 0, 0, colors='black', linestyles='-', label='equal')
    plt.legend(frameon=False, fontsize=12)
    # define limits
    if not xlim: xlim = (min(x_corr), max(x_corr))
    if not ylim: ylim = (min(y_corr), max(y_corr))
    # add fits
    if dofit:
        axScatter.plot(np.unique(x_corr), np.poly1d(np.polyfit(x_corr, y_corr, 1))(np.unique(x_corr)), 'k--', label='all')
        axScatter.plot(np.unique(x_best), np.poly1d(np.polyfit(x_best, y_best, 1))(np.unique(x_best)), 'r-', label='best')
    # add x=y
    axScatter.plot(xlim,ylim,'k-')
    # set limits
    axScatter.set_xlim(xlim)
    axScatter.set_ylim(ylim)
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    # set scales
    axScatter.set_xscale(xscale)
    axScatter.set_yscale(yscale)
    axHistx.set_xscale(xscale)
    axHisty.set_yscale(yscale)
    # no labels
    nullfmt = NullFormatter()  # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHistx.yaxis.set_major_formatter(nullfmt)
    axHisty.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    return axScatter


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))