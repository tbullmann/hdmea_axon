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

    # Structural and functional connectivity
    structural_strength, structural_delay, functional_strength, functional_delay, _, _ = Experiment(FIGURE_CULTURE).networks()

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 13))
    if not figpath:
        fig.suptitle(figurename + ' Estimate structural connectivity', fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.05)

    ax1 = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
    plot_neuron_pair(ax1, pos, axon_delay, dendrite_peak, neuron_pos, FIGURE_CONNECTED_NEURON, FIGURE_NEURON, structural_delay[FIGURE_NEURON, FIGURE_CONNECTED_NEURON], color='blue')
    mea_axes(ax1)
    ax1.set_title ('%d $\longrightarrow$ %d' % (FIGURE_NEURON, FIGURE_CONNECTED_NEURON))
    plt.title('b', loc='left', fontsize=18)

    ax2 = plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=1)
    plot_neuron_pair(ax2, pos, axon_delay, dendrite_peak, neuron_pos, FIGURE_NOT_STRUCTURAL_CONNECTED_NEURON, FIGURE_NEURON, np.NaN, color='blue')
    mea_axes(ax2)
    ax2.set_title('%d $\dashrightarrow$ %d' % (FIGURE_NEURON, FIGURE_NOT_STRUCTURAL_CONNECTED_NEURON))
    plt.title('c', loc='left', fontsize=18)

    # Schema
    ax = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=1)
    import matplotlib.image as mpimg
    img = mpimg.imread('data/sketch_structural.png')
    plt.imshow(img)
    plt.axis('off')
    plt.title('a', loc='left', fontsize=18)
    ax.set_anchor('W')
    adjust_position(ax, yshift=-0.01)


    structural_delays = list()
    for culture in FIGURE_CULTURES:
        _, structural_delay, _, functional_delay, _, _ = Experiment(culture).networks()
        structural_delays.append(list(structural_delay.values()))

    ax7 = plt.subplot2grid((4,4), (2,0), colspan=1, rowspan=2)
    plot_delays(ax7, 0, structural_delays, fill_color='blue')
    ax7.set_xlabel(r'$\mathsf{\tau_{axon}\ [ms]}$', fontsize=14)
    adjust_position(ax7, yshrink=0.02)
    plt.title('d', loc='left', fontsize=18)

    # all graphs
    plot_culture_structural_graph(1, (2, 1), pos)
    plt.title('e', loc='left', fontsize=18)
    plot_culture_structural_graph(2, (2, 2), pos)
    plot_culture_structural_graph(3, (2, 3), pos)
    plot_culture_structural_graph(4, (3, 1), pos)
    plot_culture_structural_graph(5, (3, 2), pos)
    # plot_culture_structural_graph(6, (3, 3), pos)


    show_or_savefig(figpath, figurename)


def plot_culture_structural_graph(culture, grid_pos, pos):
    _, structural_delay, _, _, _, _ = Experiment(culture).networks()
    trigger, _, axon_delay, dendrite_peak = Experiment(culture=culture).compartments()
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)
    neuron_dict = unique_neurons(structural_delay)
    axc1 = plt.subplot2grid((4, 4), grid_pos, colspan=1, rowspan=1)
    plot_network(axc1, structural_delay, neuron_pos, color='blue')
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
    make_figure(os.path.basename(__file__))

