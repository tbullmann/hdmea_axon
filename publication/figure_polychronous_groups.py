import logging
import os
import pickle

from matplotlib import pyplot as plt

from hana.polychronous import filter, combine, group, plot_pcg_on_network, plot_pcg, shuffle_network, plot_pcgs
from hana.recording import load_positions, load_timeseries, partial_timeseries
from hana.segmentation import load_compartments, load_neurites, neuron_position_from_trigger_electrode
from hana.structure import all_overlaps
from hana.function import timeseries_to_surrogates, all_peaks
from hana.plotting import plot_neuron_points, plot_network, mea_axes
from hana.misc import unique_neurons

from publication.plotting import FIGURE_ARBORS_FILE, plot_loglog_fit, FIGURE_EVENTS_FILE, adjust_position, show_or_savefig, correlate_two_dicts_verbose, without_spines_and_ticks

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


def make_figure(figurename, thr=1, figpath=None):
    """
    :param figurename:how to name the figure
    :param thr: threshold for functional connectivity (default = 1)
    :param figpath: where to save the figure
    """

    if not os.path.isfile('temp/all_delays.p'):
        axon_delay, dendrite_peak = load_neurites (FIGURE_ARBORS_FILE)
        _, _, structural_delays = all_overlaps(axon_delay, dendrite_peak)
        timelags, std_score_dict, timeseries_hist_dict = pickle.load(open('temp/standardscores.p', 'rb'))
        _, functional_delays, _, _ = all_peaks(timelags, std_score_dict, thr=thr, direction='forward')
        axonal_delays, spike_timings, pairs = correlate_two_dicts_verbose(structural_delays, functional_delays)
        putative_delays = {pair: timing for delay, timing, pair in zip(axonal_delays, spike_timings, pairs)
                           if timing-delay > 1}  # if synapse_delay = spike_timing - axonal_delay > 1ms
        pickle.dump(putative_delays, open('temp/all_delays.p', 'wb'))

    if not os.path.isfile('temp/partial_timeseries.p'):
        timeseries = load_timeseries(FIGURE_EVENTS_FILE)
        timeseries = partial_timeseries(timeseries)  # using only first 10% of recording
        pickle.dump(timeseries, open('temp/partial_timeseries.p', 'wb'))

    if not os.path.isfile('temp/connected_events.p'):
        putative_delays = pickle.load(open('temp/all_delays.p', 'rb'))
        timeseries = pickle.load(open('temp/partial_timeseries.p', 'rb'))

        connected_events = filter(timeseries, putative_delays, additional_synaptic_delay=0, synaptic_jitter=0.0005)
        pickle.dump(connected_events, open('temp/connected_events.p', 'wb'))

    # original time series on surrogate networks = surrogate 1 and 2
    if not os.path.isfile('temp/connected_events_surrogate_1.p'):
        putative_delays = pickle.load(open('temp/all_delays.p', 'rb'))
        timeseries = pickle.load(open('temp/partial_timeseries.p', 'rb'))
        surrogate_delays = shuffle_network(putative_delays, method='shuffle in-nodes')
        connected_surrogate_events = filter(timeseries, surrogate_delays, additional_synaptic_delay=0, synaptic_jitter=0.0005)
        pickle.dump(connected_surrogate_events, open('temp/connected_events_surrogate_1.p', 'wb'))
    if not os.path.isfile('temp/connected_events_surrogate_2.p'):
        putative_delays = pickle.load(open('temp/all_delays.p', 'rb'))
        timeseries = pickle.load(open('temp/partial_timeseries.p', 'rb'))
        surrogate_delays = shuffle_network(putative_delays, method='shuffle values')
        connected_surrogate_events = filter(timeseries, surrogate_delays, additional_synaptic_delay=0,
                                        synaptic_jitter=0.0005)
        pickle.dump(connected_surrogate_events, open('temp/connected_events_surrogate_2.p', 'wb'))

    # original surrogate time series on original network = surrogate 3
    if not os.path.isfile('temp/connected_events_surrogate_3.p'):

        if os.path.isfile('temp/partial_surrogate_timeseries.p'):
            timeseries = pickle.load(open('temp/partial_timeseries.p', 'rb'))
            surrogate_timeseries = timeseries_to_surrogates(timeseries, n=1, factor=2)

            # keeping only the first of several surrogate times series for each neuron
            surrogate_timeseries = { neuron: timeseries[0] for neuron, timeseries in surrogate_timeseries.items() }

            pickle.dump(surrogate_timeseries, open('temp/partial_surrogate_timeseries.p', 'wb'))

        putative_delays = pickle.load(open('temp/all_delays.p', 'rb'))
        surrogate_timeseries = pickle.load(open('temp/partial_surrogate_timeseries.p', 'rb'))
        connected_surrogate_events = filter(surrogate_timeseries, putative_delays, additional_synaptic_delay=0, synaptic_jitter=0.0005)
        pickle.dump(connected_surrogate_events, open('temp/connected_events_surrogate_3.p', 'wb'))

    # Get Polychronous groups from original data
    pcgs, pcgs_size = extract_pcgs('temp/connected_events.p')
    pcgs1, pcgs1_size = extract_pcgs('temp/connected_events_surrogate_1.p')
    pcgs2, pcgs2_size = extract_pcgs('temp/connected_events_surrogate_2.p')
    pcgs3, pcgs3_size = extract_pcgs('temp/connected_events_surrogate_3.p')

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
    polychronous_group = pcgs[2]   # 2, 3    ; 8 too many; 0, 9 interesting
    plot_pcg(ax3, polychronous_group)
    plt.ylim((0,60))
    without_spines_and_ticks(ax3)
    adjust_position(ax3, xshrink=0.01)
    xlim = ax3.get_xlim()
    ax3.text(xlim[0], 60, r'  $\bullet$  spikes ', fontsize=14,
             horizontalalignment='left', verticalalignment='top')
    ax3.text(xlim[0], 55, r'  $\leftrightarrows$ polychronous group', color='r', fontsize=14,
                horizontalalignment='left', verticalalignment='top')
    plt.title('c', loc='left', fontsize=18)

    # plot example of a single polychronous group onto network
    ax4 = plt.subplot(224)
    trigger, _, axon_delay, dendrite_peak = load_compartments(FIGURE_ARBORS_FILE)
    pos = load_positions()
    neuron_pos = neuron_position_from_trigger_electrode (pos, trigger)
    all_delay = pickle.load(open('temp/all_delays.p', 'rb'))
    plot_pcg_on_network(ax4, polychronous_group, all_delay, neuron_pos)
    plt.title('d', loc='left', fontsize=18)
    ax4.text(300, 150, '$\leftrightarrows$ putative chemical synapses', color='gray', fontsize=14)
    ax4.text(300, 300, '$\leftrightarrows$ activated by polychronous group', color='r', fontsize=14)

    show_or_savefig(figpath, figurename)


def extract_pcgs(filename):
    logging.info('Load data...')
    connected_events = pickle.load(open(filename, 'rb'))
    graph_of_connected_events = combine(connected_events)
    logging.info('Data: %d events form %d pairs'
                 % (len(graph_of_connected_events.nodes()), len(graph_of_connected_events.edges())))
    list_of_polychronous_groups = group(graph_of_connected_events)
    logging.info('Data: Forming %d polycronous groups' % len(list_of_polychronous_groups))
    polychronous_group_size = list(len(g) for g in list_of_polychronous_groups)
    return list_of_polychronous_groups, polychronous_group_size


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
    trigger, _, axon_delay, dendrite_peak = load_compartments(FIGURE_ARBORS_FILE)
    pos = load_positions()
    neuron_pos = neuron_position_from_trigger_electrode (pos, trigger)

    # get example networks
    original_network = pickle.load(open('temp/all_delays.p', 'rb'))
    network_with_shuffle_in_nodes = shuffle_network(original_network, method='shuffle in-nodes')
    network_with_shuffle_values = shuffle_network(original_network, method='shuffle values')

    plot_small_networks(neuron_pos, original_network, network_with_shuffle_in_nodes, network_with_shuffle_values)


if __name__ == "__main__":
    # testing_algorithm()
    # test_plot_surrogates()
    # plt.show()
    make_figure(os.path.basename(__file__))
