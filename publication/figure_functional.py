from __future__ import division

import logging
import os

import numpy as np
from matplotlib import pyplot as plt

from hana.function import timelag_standardscore, all_peaks
from hana.misc import unique_neurons
from hana.plotting import plot_neuron_points, plot_neuron_id, plot_neuron_pair, plot_network, mea_axes, highlight_connection, \
    plot_timeseries_hist_and_surrogates
from hana.recording import load_positions, average_electrode_area
from hana.segmentation import neuron_position_from_trigger_electrode
from hana.structure import find_overlap, all_overlaps
from misc.figure_structural import plot_two_colorbars
from publication.data import Experiment, FIGURE_CULTURE, FIGURE_NEURON, FIGURE_CONNECTED_NEURON, \
    FIGURE_NOT_FUNCTIONAL_CONNECTED_NEURON, FIGURE_NOT_STRUCTURAL_CONNECTED_NEURON, FIGURE_THRESHOLD_OVERLAP_AREA
from hana.plotting import plot_std_score_and_peaks
from publication.plotting import show_or_savefig, adjust_position, without_spines_and_ticks
from publication.figure_effective import plot_delays
from publication.data import FIGURE_CULTURES

logging.basicConfig(level=logging.DEBUG)


# Final version

def make_figure(figurename, figpath=None):

    # Neuron compartments and AIS positions
    trigger, _, axon_delay, dendrite_peak = Experiment(FIGURE_CULTURE).compartments()
    pos = load_positions(mea='hidens')
    # electrode_area = average_electrode_area(pos)
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)

    # functional connectivity
    _, _, functional_strength, functional_delay, _, _ = Experiment(FIGURE_CULTURE).networks()
    # Making figure
    fig = plt.figure(figurename, figsize=(13, 13))
    if not figpath:
        fig.suptitle(figurename + ' Estimate functional connectivity', fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.05)


    # Examples for functional connected and unconnected neurons and functional network
    timeseries = Experiment(FIGURE_CULTURE).timeseries()
    timeseries_surrogates = Experiment(FIGURE_CULTURE).timeseries_surrogates()
    timelags, std_score_dict, timeseries_hist_dict = Experiment(FIGURE_CULTURE).standardscores()
    peak_score, peak_timelag, z_threshold = all_peaks(timelags, std_score_dict)

    # Schema
    ax = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=1)
    import matplotlib.image as mpimg
    img = mpimg.imread('data/sketch_functional.png')
    plt.imshow(img)
    plt.axis('off')
    plt.title('a', loc='left', fontsize=18)
    ax.set_anchor('W')
    adjust_position(ax, yshift=-0.01)


    ax1 = plt.subplot2grid((4,3), (0,1), colspan=1, rowspan=1)
    # Calculate (again) details for a single neuron pair
    timelags, std_score, timeseries_hist, surrogates_mean, surrogates_std \
        = timelag_standardscore(timeseries[FIGURE_NEURON], timeseries[FIGURE_CONNECTED_NEURON], timeseries_surrogates[FIGURE_CONNECTED_NEURON])
    peak_score, peak_timelag, _ = all_peaks(timelags, std_score_dict)  # thr=thr, direction=direction)
    # Plot histograms for single neuron pair
    plot_timeseries_hist_and_surrogates(ax1, timelags, timeseries_hist, surrogates_mean, surrogates_std, loc=None)
    ax1.set_xlabel(r'$\mathsf{time\ lag\ \Delta t\ [ms]}$', fontsize = 14)

    ax1.set_title ('%d $\longrightarrow$ %d' % (FIGURE_NEURON, FIGURE_CONNECTED_NEURON))
    without_spines_and_ticks(ax1)
    plt.title('b', loc='left', fontsize=18)
    ax1.set_xlim((0, 5))

    ax2 = plt.subplot2grid((4,3), (1,1), colspan=1, rowspan=1)
    plot_z_score (ax2, FIGURE_NEURON, FIGURE_CONNECTED_NEURON, z_threshold, peak_timelag, timelags, std_score_dict)
    without_spines_and_ticks(ax2)

    adjust_position(ax1, yshrink=0.02, yshift=+0.01, xshift=-0.02)
    adjust_position(ax2, yshrink=0.02, yshift=+0.03, xshift=-0.02)
    ax2.set_ylim((-5, 15))
    ax2.set_xlabel(r'$\mathsf{time\ lag\ \Delta t\ [ms]}$', fontsize = 14)

    # Not connected neuron pair

    ax3 = plt.subplot2grid((4, 3), (0, 2), colspan=1, rowspan=1)
    # Calculate (again) details for a single neuron pair
    timelags, std_score, timeseries_hist, surrogates_mean, surrogates_std \
        = timelag_standardscore(timeseries[FIGURE_NEURON], timeseries[FIGURE_NOT_FUNCTIONAL_CONNECTED_NEURON], timeseries_surrogates[FIGURE_NOT_FUNCTIONAL_CONNECTED_NEURON])
    peak_score, peak_timelag, _ = all_peaks(timelags, std_score_dict)  # thr=thr, direction=direction)
    # Plot histograms for single neuron pair
    plot_timeseries_hist_and_surrogates(ax3, timelags, timeseries_hist, surrogates_mean, surrogates_std, loc=None)

    ax3.set_title ('%d $\longrightarrow$ %d' % (FIGURE_NEURON, FIGURE_NOT_FUNCTIONAL_CONNECTED_NEURON))
    without_spines_and_ticks(ax3)
    plt.title('c', loc='left', fontsize=18)
    ax3.set_xlim((0, 5))
    ax3.set_xlabel(r'$\mathsf{time\ lag\ \Delta t\ [ms]}$', fontsize = 14)

    ax4 = plt.subplot2grid((4, 3), (1, 2), colspan=1, rowspan=1)
    plot_z_score (ax4, FIGURE_NEURON, FIGURE_NOT_FUNCTIONAL_CONNECTED_NEURON, z_threshold, peak_timelag, timelags, std_score_dict)
    without_spines_and_ticks(ax4)

    adjust_position(ax3, yshrink=0.02, yshift=+0.01)
    adjust_position(ax4, yshrink=0.02, yshift=+0.03)
    ax4.set_ylim((-5, 15))
    ax4.set_xlabel(r'$\mathsf{time\ lag\ \Delta t\ [ms]}$', fontsize = 14)

    # Summary
    functional_delays = list()
    for culture in FIGURE_CULTURES:
        _, _, _, functional_delay, _, _ = Experiment(culture).networks()
        functional_delays.append(list(functional_delay.values()))

    ax8 = plt.subplot2grid((4,4), (2,0), colspan=1, rowspan=2)
    print(functional_delays)
    plot_delays(ax8, 0, functional_delays, fill_color='red')
    ax8.set_xlabel(r'$\mathsf{\tau_{spike}\ [ms]}$', fontsize=14)
    adjust_position(ax8, yshrink=0.02)
    plt.title('d', loc='left', fontsize=18)

    # all graphs
    plot_culture_functional_graph(1, (2, 1), pos)
    plt.title('e', loc='left', fontsize=18)
    plot_culture_functional_graph(2, (2, 2), pos)
    plot_culture_functional_graph(3, (2, 3), pos)
    plot_culture_functional_graph(4, (3, 1), pos)
    plot_culture_functional_graph(5, (3, 2), pos)
    # plot_culture_functional_graph(6, (3, 3), pos)


    show_or_savefig(figpath, figurename)


def print_functional_for_figure_neuron():
    timeseries = Experiment(FIGURE_CULTURE).timeseries()
    timelags, std_score_dict, timeseries_hist_dict = Experiment(FIGURE_CULTURE).standardscores()
    peak_score, peak_timelag, z_threshold = all_peaks(timelags, std_score_dict)

    print(z_threshold)
    for post in timeseries.keys():
        try:
            print("%d, %d have %1.3f ms delay with score of %1.3f" % (
            FIGURE_NEURON, post, peak_timelag[(FIGURE_NEURON, post)], peak_score[(FIGURE_NEURON, post)]))
        except:
            print("%d, %d are not connected" % (FIGURE_NEURON, post))
    print(peak_timelag[(5, 10)])


def plot_culture_functional_graph(culture, grid_pos, pos):
    trigger, _, axon_delay, dendrite_peak = Experiment(culture=culture).compartments()
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)
    _, _, _, functional_delay, _, _ = Experiment(culture).networks()
    neuron_dict = unique_neurons(functional_delay)
    axc1 = plt.subplot2grid((4, 4), grid_pos, colspan=1, rowspan=1)
    plot_network(axc1, functional_delay, neuron_pos, color='red')
    plot_neuron_points(axc1, neuron_dict, neuron_pos)
    mea_axes(axc1)
    axc1.set_title ('culture %d' % culture)
    adjust_position(axc1, yshrink=0.02)


def plot_z_score (ax3, pre, post, thr, peak_timelag, timelags, std_score_dict):
    std_score = std_score_dict[(pre, post)]
    if (pre, post) in peak_timelag:
        peak = peak_timelag[(pre, post)]
    else:
        peak = None
    plot_std_score_and_peaks(ax3, timelags, std_score, thr=thr, peak=peak, loc=0)
    ax3.set_xlim((0,5))

if __name__ == "__main__":
    # print_functional_for_figure_neuron()  # for finding the neuron pairs in the figure
    make_figure(os.path.basename(__file__))

