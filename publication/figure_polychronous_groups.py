import logging
import os

from matplotlib import pyplot as plt

from hana.misc import unique_neurons
from hana.plotting import plot_neuron_points, plot_network, mea_axes
from hana.polychronous import plot_pcg_on_network, plot_pcg, shuffle_network
from hana.recording import load_positions
from hana.segmentation import neuron_position_from_trigger_electrode

from publication.data import Experiment, FIGURE_CULTURE
from publication.plotting import plot_loglog_fit, adjust_position, \
    show_or_savefig, correlate_two_dicts_verbose, without_spines_and_ticks

logging.basicConfig(level=logging.DEBUG)


import numpy as np


def testing_algorithm():
    source = np.array([6, 14, 25, 55, 70, 80])
    shifted_source = source + 5
    jitter = 2
    target = np.array([10, 20, 60, 200, 300, 400, 500, 600, 700])
    for offset in (0, 1):
        position = (np.searchsorted(target, shifted_source)-offset).clip(0, len(target)-1)
        print (position)
        print (target[position])
        print (shifted_source)
        print (shifted_source-target[position])
        valid = np.abs(shifted_source-target[position])<jitter
        print (valid)
        valid_source = source[valid]
        valid_target = target[position[valid]]
        print (zip(valid_source, valid_target))
        print (sum(valid))


def make_figure(figurename, figpath=None):
    """
    :param figurename:how to name the figure
    :param thr: threshold for functional connectivity (default = 1)
    :param figpath: where to save the figure
    """

    pcgs, _ = Experiment(FIGURE_CULTURE).polychronous_groups()
    pcgs_size, pcgs1_size, pcgs2_size, pcgs3_size = Experiment(FIGURE_CULTURE).polychronous_group_sizes_with_surrogates()

    # Making figure
    fig = plt.figure(figurename, figsize=(12, 10))
    if not figpath:
        fig.suptitle(figurename + ' Polychronous groups', fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.05)

    # # Plot polychronous groups with in different arrow colors
    # ax = plt.subplot(221)
    # plot_pcgs(ax, list_of_polychronous_groups[35:55])

    # # plot example of a single polychronous group
    # ax1 = plt.subplot(221)
    # plot(graph_of_connected_events)
    # plt.xlim((590,700))
    # plt.ylim((0,60))
    # adjust_position(ax1, xshrink=0.01)
    # plt.title('a', loc='left', fontsize=18)

    panel_explaining_surrogate_networks()

    # plot size distribution
    ax2 = plt.subplot(222)
    plot_loglog_fit(ax2, pcgs_size, fit=False)
    plot_loglog_fit(ax2, pcgs1_size, datamarker='r-', datalabel = 'surrogate network 1', fit=False)
    plot_loglog_fit(ax2, pcgs2_size, datamarker='g-', datalabel = 'surrogate network 2', fit=False)
    plot_loglog_fit(ax2, pcgs3_size, datamarker='b-', datalabel = 'surrogate timeseries', fit=False)
    ax2.set_ylabel('size [number of spikes / polychronous group]')
    without_spines_and_ticks(ax2)
    ax2.set_ylim((1, 1000))
    ax2.set_xlim((1, 13000))
    plt.title('b', loc='left', fontsize=18)

    # plot example of a single polychronous group with arrows
    ax3 = plt.subplot(223)
    polychronous_group = pcgs[1]   # 3 has allmost all neurons involved
    plot_pcg(ax3, polychronous_group)
    plt.ylim((0,60))
    without_spines_and_ticks(ax3)
    adjust_position(ax3, xshrink=0.01)
    xlim = ax3.get_xlim()
    ax3.text(xlim[0], 60, r'  $\bullet$  spikes ', fontsize=14,
             horizontalalignment='left', verticalalignment='top')
    ax3.text(xlim[0], 55, r'  $\leftrightarrows$ spike pattern of polychronous group', color='r', fontsize=14,
                horizontalalignment='left', verticalalignment='top')
    plt.title('c', loc='left', fontsize=18)

    # plot example of a single polychronous group onto network
    ax4 = plt.subplot(224)
    trigger, _, axon_delay, dendrite_peak = Experiment(FIGURE_CULTURE).compartments()
    pos = load_positions()
    neuron_pos = neuron_position_from_trigger_electrode (pos, trigger)
    all_delay = Experiment(FIGURE_CULTURE).putative_delays()
    plot_pcg_on_network(ax4, polychronous_group, all_delay, neuron_pos)
    plt.title('d', loc='left', fontsize=18)
    ax4.text(300, 150, '$\leftrightarrows$ putative chemical synapses', color='gray', fontsize=14)
    ax4.text(300, 300, '$\leftrightarrows$ activated polychronous group', color='r', fontsize=14)

    show_or_savefig(figpath, figurename)


def plot_small_networks(neuron_pos, original_network, network_with_shuffle_in_nodes, network_with_shuffle_values):

    original_values, shuffled_values, _ = correlate_two_dicts_verbose(original_network, network_with_shuffle_values)

    ax1 = plt.subplot(441)
    ax2 = plt.subplot(442)
    ax3 = plt.subplot(445)
    ax4 = plt.subplot(446)

    plot_small_network(ax1, original_network, neuron_pos)
    ax1.set_title('a', loc='left', fontsize=18)
    ax1.text(100,400, 'original network', fontsize=14)
    adjust_position(ax1, xshift=-0.02)

    plot_small_network(ax2, network_with_shuffle_in_nodes, neuron_pos)
    ax2.text(100,400, 'surrogate network 1\nwith shuffled post-\nsynaptic neurons', fontsize=14)
    adjust_position(ax2, xshift=-0.02)

    plot_small_network(ax3, network_with_shuffle_values, neuron_pos)
    ax3.text(100,400, 'surrogate network 2\nwith shuffled delays', fontsize=14)
    adjust_position(ax3, xshift=-0.02)

    ax4.plot( original_values, shuffled_values, color='k', marker='.', linestyle='none')
    ax4.set_xlabel(r'$\mathsf{original\ \tau_{spike}\ [ms]}$', fontsize=14)
    ax4.set_ylabel(r'$\mathsf{surrogate\ \tau_{spike}\ [ms]}$', fontsize=14)
    ax4.set_xlim((0, 5))
    ax4.set_ylim((0, 5))
    without_spines_and_ticks(ax4)
    adjust_position(ax4, yshrink=0.02, xshrink=0.03, xshift=-0.02, yshift=0.01)

    arrow_outside(ax1, xy=(1.7, 0.5), color='r')
    arrow_outside(ax1, xy=(1.3, -0.3), color='g')


def arrow_outside(ax1, xy=(2., 0.5), color='r'):
    ax1.annotate('', xy=xy, xycoords='axes fraction', xytext=(0.5, 0.4),
                 textcoords='axes fraction', size=20,
                 arrowprops=dict(color=color, arrowstyle="->", linewidth=2, connectionstyle="arc3,rad=-0.3"))


def plot_small_network(ax, network, neuron_pos):
    plot_neuron_points(ax, unique_neurons(network), neuron_pos)
    plot_network(ax, network, neuron_pos, color='gray')
    mea_axes(ax, style='None')


def panel_explaining_surrogate_networks():

    # neuron_pos
    trigger, _, axon_delay, dendrite_peak = Experiment(FIGURE_CULTURE).compartments()
    pos = load_positions()
    neuron_pos = neuron_position_from_trigger_electrode (pos, trigger)

    # get example networks
    original_network = Experiment(FIGURE_CULTURE).putative_delays()
    network_with_shuffle_in_nodes = shuffle_network(original_network, method='shuffle in-nodes')
    network_with_shuffle_values = shuffle_network(original_network, method='shuffle values')

    plot_small_networks(neuron_pos, original_network, network_with_shuffle_in_nodes, network_with_shuffle_values)


if __name__ == "__main__":
    # testing_algorithm()
    # test_plot_surrogates()
    # plt.show()
    make_figure(os.path.basename(__file__))
