from __future__ import division
import logging
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from hana.function import timeseries_to_surrogates, all_timelag_standardscore, all_peaks
from hana.recording import load_timeseries, average_electrode_area
from hana.segmentation import load_neurites
from hana.structure import all_overlaps
from publication.plotting import show_or_savefig, FIGURE_ARBORS_FILE, FIGURE_EVENTS_FILE, TEMPORARY_PICKELED_NETWORKS, \
    plot_parameter_dependency, format_parameter_plot, compare_networks, analyse_networks, label_subplot, adjust_position, \
    correlate_two_dicts, kernel_density, axes_to_3_axes

logging.basicConfig(level=logging.DEBUG)

# Data preparation

def maybe_get_functional_and_structural_networks(arbors_filename, events_filename, output_pickel_name):
    if os.path.isfile(output_pickel_name): return

    # structural
    axon_delay, dendrite_peak = load_neurites (arbors_filename)
    structural_strength, _, structural_delay = all_overlaps(axon_delay, dendrite_peak, thr_peak=0, thr_ratio=0, thr_overlap=1)

    # functional: only use forward direction
    timeseries = load_timeseries(events_filename)
    logging.info('Surrogate data')
    timeseries_surrogates = timeseries_to_surrogates(timeseries, n=20, factor=2)
    logging.info('Compute standard score for histograms')
    timelags, std_score_dict, timeseries_hist_dict = all_timelag_standardscore(timeseries, timeseries_surrogates)
    functional_strength, functional_delay, _, _ = all_peaks(timelags, std_score_dict, thr=1, direction='forward')

    pickle.dump((structural_strength, structural_delay, functional_strength, functional_delay), open(output_pickel_name, 'wb'))
    logging.info('Saved data')


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
    structural_thresholds, functional_thresholds, intersection_size, structural_index, jaccard_index, functional_index\
        = compare_networks(structural_strengths, functional_strengths, scale='log')
    plot_parameter_dependency(ax3, structural_index, structural_thresholds, functional_thresholds,
                              fmt='%1.1f', levels=np.linspace(0, 1, 11))
    format_parameter_plot(ax3)
    # ax3.text(3,120, r'Structural index ${|F \cup S|}/{|S|}$', size=15)
    ax3.set_title(r'$\mathsf{i_S={|F \cup S|}/{|S|}}$', fontsize=14)
    adjust_position(ax3, xshrink=0.02, yshrink=0.02, xshift=0.02, yshift=0.01)

    # plot distribution of strength and delays
    ax4 = plt.subplot(223)
    axScatter1 = plot_correlation(ax4, structural_strengths, functional_strengths, xscale='log', yscale='log')
    axScatter1.set_xlabel (r'$\mathsf{|A \cup D|\ [\mu m^2}$]', fontsize=14)
    axScatter1.set_ylabel (r'$\mathsf{z_{max}}$', fontsize=14)

    ax5 = plt.subplot(224)
    axScatter2 = plot_correlation(ax5, structural_delays, functional_delays, xlim=(0,5), ylim=(0,5))
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