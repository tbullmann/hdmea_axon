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
from publication.figure_synapses import plot_delays
from publication.data import FIGURE_CULTURES

logging.basicConfig(level=logging.DEBUG)


# Final version

def make_figure(figurename, figpath=None):

    # Neuron compartments and AIS positions
    trigger, _, axon_delay, dendrite_peak = Experiment(FIGURE_CULTURE).compartments()
    pos = load_positions(mea='hidens')
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)

    # functional connectivity
    _, structural_delay, _, functional_delay, _, _ = Experiment(FIGURE_CULTURE).networks()

    # (5, 35) 1.28881163085 0.15
    PRE_SYNAPTIC_NEURON = 5
    POST_SYNAPTIC_NEURON = 23   #22,23,27,31

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 13))
    if not figpath:
        fig.suptitle(figurename + ' Compare functional with structural connectivity', fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.05)


    # Functional

    timeseries = Experiment(FIGURE_CULTURE).timeseries()
    timeseries_surrogates = Experiment(FIGURE_CULTURE).timeseries_surrogates()
    timelags, std_score_dict, timeseries_hist_dict = Experiment(FIGURE_CULTURE).standardscores()
    peak_score, peak_timelag, z_threshold = all_peaks(timelags, std_score_dict)


    ax3 = plt.subplot2grid((4,3), (0,0), colspan=1, rowspan=1)
    # Calculate (again) details for a single neuron pair
    timelags, std_score, timeseries_hist, surrogates_mean, surrogates_std \
        = timelag_standardscore(timeseries[PRE_SYNAPTIC_NEURON], timeseries[POST_SYNAPTIC_NEURON], timeseries_surrogates[POST_SYNAPTIC_NEURON])
    peak_score, peak_timelag, _ = all_peaks(timelags, std_score_dict)  # thr=thr, direction=direction)
    # Plot histograms for single neuron pair
    plot_timeseries_hist_and_surrogates(ax3, timelags, timeseries_hist, surrogates_mean, surrogates_std, loc=None)

    ax3.set_title ('%d $\longrightarrow$ %d' % (PRE_SYNAPTIC_NEURON, POST_SYNAPTIC_NEURON))
    without_spines_and_ticks(ax3)
    ax3.set_xlabel(r'$\mathsf{time\ lag\ \Delta t\ [ms]}$', fontsize = 14)
    plt.title('a', loc='left', fontsize=18)
    ax3.set_xlim((0, 5))

    ax4 =  plt.subplot2grid((4,3), (1,0), colspan=1, rowspan=1)
    plot_z_score (ax4, PRE_SYNAPTIC_NEURON, POST_SYNAPTIC_NEURON, z_threshold, peak_timelag, timelags, std_score_dict)
    without_spines_and_ticks(ax4)

    adjust_position(ax3, yshrink=0.03, yshift=-0.015)
    adjust_position(ax4, yshrink=0.03, yshift=+0.01)
    ax4.set_xlabel(r'$\mathsf{time\ lag\ \Delta t\ [ms]}$', fontsize = 14)
    ax4.set_ylim((-4, 10))


    # Structural

    ax1 = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
    plot_neuron_pair(ax1, pos, axon_delay, dendrite_peak, neuron_pos, POST_SYNAPTIC_NEURON, PRE_SYNAPTIC_NEURON, structural_delay[PRE_SYNAPTIC_NEURON, POST_SYNAPTIC_NEURON], color='blue')
    mea_axes(ax1)
    ax1.set_title ('%d $\longrightarrow$ %d' % (PRE_SYNAPTIC_NEURON, POST_SYNAPTIC_NEURON))
    plt.title('b', loc='left', fontsize=18)


    # Functional vs structural delays

    ax5 = plt.subplot2grid((4,3), (0,2), colspan=1, rowspan=2)

    _, structural_delay, _, functional_delay, _, _ = Experiment(FIGURE_CULTURE).networks()
    x = list()
    y = list()
    for pair in structural_delay:
        if pair in functional_delay:
            x.append(structural_delay[pair])
            y.append(functional_delay[pair])

    ax5.scatter(x, y, color='black')
    ax5.plot([0,3],[0,3],color='blue')
    ax5.fill([0,3,3,0], [0,3,0,0], fill=False, hatch='\\', linewidth=0)
    ax5.set_xlim(0, 3)
    ax5.set_ylim(0, 5)
    ax5.set_aspect('equal', adjustable='box')
    ax5.set_xlabel(r'$\mathsf{\tau_{axon}\ [ms]}$', fontsize=14)
    ax5.set_ylabel(r'$\mathsf{\tau_{spike}\ [ms]}$', fontsize=14)
    ax5.set_title ('culture %d' % FIGURE_CULTURE)
    adjust_position(ax5, yshrink=0.042)
    without_spines_and_ticks(ax5)
    plt.title('c', loc='left', fontsize=18)

    # Summary
    synaptic_delays = list()
    for culture in FIGURE_CULTURES:
        _, structural_delay, _, functional_delay, _, _ = Experiment(culture).networks()
        differences = list()
        for pair in structural_delay:
            if pair in functional_delay:
                differences.append(functional_delay[pair]-structural_delay[pair])
        synaptic_delays.append(differences)

    ax8 = plt.subplot2grid((4,4), (2,0), colspan=1, rowspan=2)
    plot_delays(ax8, -4, synaptic_delays, xticks = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    ax8.set_xlabel(r'$\mathsf{\tau_{spike}-\tau_{axon}\ [ms]}$', fontsize=14)
    adjust_position(ax8, yshrink=0.02)
    plt.title('d', loc='left', fontsize=18)

    # Sche,a
    ax = plt.subplot2grid((4,4), (2,1), colspan=3, rowspan=2)
    import matplotlib.image as mpimg
    img = mpimg.imread('figures/AnalysisMethods.png')
    plt.imshow(img)
    plt.axis('off')
    plt.title('e', loc='left', fontsize=18)
    ax.set_anchor('W')
    adjust_position(ax,yshift=-0.02)
    # adjust_position(ax, xshift=0.02, yshift=-0.03)

    show_or_savefig(figpath, figurename)


def compare_connection():

    # Neuron compartments and AIS positions
    trigger, _, axon_delay, dendrite_peak = Experiment(FIGURE_CULTURE).compartments()
    pos = load_positions(mea='hidens')
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)

    # functional connectivity
    _, structural_delay, _, functional_delay, _, _ = Experiment(FIGURE_CULTURE).networks()

    for neuron_pair in structural_delay:
        if neuron_pair in functional_delay:
            if structural_delay[neuron_pair] > functional_delay[neuron_pair]:
                if neuron_pair[0]==5:
                    print neuron_pair, structural_delay[neuron_pair], functional_delay[neuron_pair]

    for neuron_pair in structural_delay:
        if neuron_pair[0]==5:
            print neuron_pair




def plot_z_score (ax3, pre, post, thr, peak_timelag, timelags, std_score_dict):
    std_score = std_score_dict[(pre, post)]
    if (pre, post) in peak_timelag:
        peak = peak_timelag[(pre, post)]
    else:
        peak = None
    plot_std_score_and_peaks(ax3, timelags, std_score, thr=thr, peak=peak, loc=0)
    ax3.set_xlim((0,5))

if __name__ == "__main__":
    compare_connection()
    make_figure(os.path.basename(__file__))

