from __future__ import division

import logging
import os

import numpy as np
from matplotlib import pyplot as plt

from hana.function import timelag_standardscore, all_peaks
from hana.misc import unique_neurons
from hana.plotting import plot_neuron_points, plot_neuron_id, plot_neuron_pair, plot_network, mea_axes, highlight_connection
from hana.recording import load_positions, average_electrode_area
from hana.segmentation import neuron_position_from_trigger_electrode
from hana.structure import find_overlap, all_overlaps
from misc.figure_structural import plot_two_colorbars
from publication.data import Experiment, FIGURE_CULTURE, FIGURE_NEURON, FIGURE_CONNECTED_NEURON, \
    FIGURE_NOT_FUNCTIONAL_CONNECTED_NEURON, FIGURE_NOT_STRUCTURAL_CONNECTED_NEURON, FIGURE_THRESHOLD_OVERLAP_AREA
from publication.figure_functional import plot_std_score_and_peaks
from publication.plotting import show_or_savefig, adjust_position, without_spines_and_ticks

logging.basicConfig(level=logging.DEBUG)


# Final version

def make_figure(figurename, figpath=None):

    # Neuron compartments and AIS positions
    trigger, _, axon_delay, dendrite_peak = Experiment(FIGURE_CULTURE).compartments()
    pos = load_positions(mea='hidens')
    # electrode_area = average_electrode_area(pos)
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)

    # Structural and functional connectivity
    structural_strength, structural_delay, functional_strength, functional_delay, _, _ = Experiment(FIGURE_CULTURE).networks()

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 14))
    if not figpath:
        fig.suptitle(figurename + ' Estimate structural and functional connectivity', fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.05)

    ax1 = plt.subplot(421)
    plot_neuron_pair(ax1, pos, axon_delay, dendrite_peak, neuron_pos, FIGURE_CONNECTED_NEURON, FIGURE_NEURON, structural_delay[FIGURE_NEURON, FIGURE_CONNECTED_NEURON])
    mea_axes(ax1, style='off')
    ax1.set_title ('neuron pair %d $\longrightarrow$ %d' % (FIGURE_NEURON, FIGURE_CONNECTED_NEURON))
    plot_two_colorbars(ax1)
    adjust_position(ax1, yshrink=0.01)
    plt.title('a', loc='left', fontsize=18)

    ax2 = plt.subplot(423)
    plot_neuron_pair(ax2, pos, axon_delay, dendrite_peak, neuron_pos, FIGURE_NOT_STRUCTURAL_CONNECTED_NEURON, FIGURE_NEURON, np.NaN)
    mea_axes(ax2, style='off')
    ax2.set_title('neuron pair %d $\dashrightarrow$ %d' % (FIGURE_NEURON, FIGURE_NOT_STRUCTURAL_CONNECTED_NEURON))
    plot_two_colorbars(ax2)
    adjust_position(ax2, yshrink=0.01)
    plt.title('b', loc='left', fontsize=18)

    # Whole network
    ax3 = plt.subplot(222)
    # all_overlap, all_ratio, all_delay = all_overlaps(axon_delay, dendrite_peak, thr_overlap=thr_overlap)
    plot_network (ax3, structural_delay, neuron_pos, color='gray')
    plot_neuron_points(ax3, unique_neurons(structural_delay), neuron_pos)
    plot_neuron_id(ax3, trigger, neuron_pos)
    highlight_connection(ax3, (FIGURE_NEURON, FIGURE_NOT_STRUCTURAL_CONNECTED_NEURON), neuron_pos, connected=False)
    highlight_connection(ax3, (FIGURE_NEURON, FIGURE_CONNECTED_NEURON), neuron_pos)
    # ax3.text(200, 250,r'$\mathsf{\rho=%3d\ \mu m^2}$' % FIGURE_THRESHOLD_OVERLAP_AREA, fontsize=18)
    mea_axes(ax3)
    plt.title('c     structural connectivity graph', loc='left', fontsize=18)

    # Examples for functional connected and unconnected neurons and functional network
    timelags, std_score_dict, timeseries_hist_dict = Experiment(FIGURE_CULTURE).standardscores()
    peak_score, peak_timelag, z_threshold = all_peaks(timelags, std_score_dict)

    # Plotting forward
    ax4 = plt.subplot(224)
    plot_network(ax4, peak_score, neuron_pos, color='gray')
    neuron_dict = unique_neurons(peak_score)
    plot_neuron_points(ax4, neuron_dict, neuron_pos)
    plot_neuron_id(ax4, neuron_dict, neuron_pos)
    highlight_connection(ax4, (FIGURE_NEURON, FIGURE_NOT_FUNCTIONAL_CONNECTED_NEURON), neuron_pos, connected=False)
    highlight_connection(ax4, (FIGURE_NEURON, FIGURE_CONNECTED_NEURON), neuron_pos)
    mea_axes(ax4)
    plt.title('f     functional connectivity graph', loc='left', fontsize=18)

    ax5 = plt.subplot(425)
    plot_z_score (ax5, FIGURE_NEURON, FIGURE_CONNECTED_NEURON, z_threshold, peak_timelag, timelags, std_score_dict)
    ax5.set_title ('neuron pair %d $\longrightarrow$ %d' % (FIGURE_NEURON, FIGURE_CONNECTED_NEURON))
    without_spines_and_ticks(ax5)
    plt.title('d', loc='left', fontsize=18)
    ax6 = plt.subplot(427)
    plot_z_score (ax6, FIGURE_NEURON, FIGURE_NOT_FUNCTIONAL_CONNECTED_NEURON, z_threshold, peak_timelag, timelags, std_score_dict)
    ax6.set_title ('neuron pair %d $\longrightarrow$ %d' % (FIGURE_NEURON, FIGURE_NOT_FUNCTIONAL_CONNECTED_NEURON))
    without_spines_and_ticks(ax6)
    plt.title('e', loc='left', fontsize=18)
    adjust_position(ax5, yshrink=0.01, xshrink=0.04)
    adjust_position(ax6, yshrink=0.01, xshrink=0.04)
    ax5.set_ylim((-5, 15))
    ax6.set_ylim((-5, 15))

    show_or_savefig(figpath, figurename)


def plot_z_score (ax3, pre, post, thr, peak_timelag, timelags, std_score_dict):
    std_score = std_score_dict[(pre, post)]
    if (pre, post) in peak_timelag:
        peak = peak_timelag[(pre, post)]
    else:
        peak = None
    plot_std_score_and_peaks(ax3, timelags, std_score, thr=thr, peak=peak, loc=0)


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))

