import logging
import os
import pickle

from matplotlib import pyplot as plt

from future_publication.polychronous import filter, combine, group, plot, plot_pcg_on_network, plot_pcg
from hana.recording import load_positions, load_timeseries, partial_timeseries
from hana.segmentation import load_compartments, load_neurites, neuron_position_from_trigger_electrode
from hana.structure import all_overlaps
from publication.plotting import FIGURE_ARBORS_FILE, plot_loglog_fit, FIGURE_EVENTS_FILE, label_subplot, shrink_axes

logging.basicConfig(level=logging.DEBUG)


# Previous versions for finding connected events

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


# Final figure 8

def figure08():
    if not os.path.isfile('temp/all_delays.p'):
        axon_delay, dendrite_peak = load_neurites (FIGURE_ARBORS_FILE)
        _, _, all_delays = all_overlaps(axon_delay, dendrite_peak)
        pickle.dump(all_delays, open('temp/all_delays.p', 'wb'))

    if not os.path.isfile('temp/partial_timeseries.p'):
        timeseries = load_timeseries(FIGURE_EVENTS_FILE)
        timeseries = partial_timeseries(timeseries)
        pickle.dump(timeseries, open('temp/partial_timeseries.p', 'wb'))

    if not os.path.isfile('temp/connected_events.p'):
        all_delays = pickle.load(open('temp/all_delays.p', 'rb'))
        timeseries = pickle.load(open('temp/partial_timeseries.p', 'rb'))

        connected_events = filter(timeseries, all_delays)
        pickle.dump(connected_events, open('temp/connected_events.p', 'wb'))

    if not os.path.isfile('temp/connected_surrogate_events.p'):
        print ('DO!')
        all_delays = pickle.load(open('temp/all_delays.p', 'rb'))
        timeseries = pickle.load(open('temp/partial_timeseries.p', 'rb'))

        def random_shuffle(a):
            keys = a.keys()
            # np.random.shuffle(keys)
            print keys
            pre, post = zip(*keys)
            new_post = list(post)
            np.random.shuffle(new_post)
            keys = zip(pre, new_post)
            print keys
            b = dict(zip(keys, a.values()))
            return b

        surrogate_delays = random_shuffle(all_delays)
        connected_events = filter(timeseries, surrogate_delays)
        pickle.dump(connected_events, open('temp/connected_surrogate_events.p', 'wb'))


    connected_events = pickle.load(open('temp/connected_events.p', 'rb'))
    connected_surrogate_events = pickle.load(open('temp/connected_surrogate_events.p', 'rb'))
    # TODO: Add plot of surrogate polychronous groups
    graph_of_connected_events = combine(connected_events)
    list_of_polychronous_groups = group(graph_of_connected_events)

    # for debugging:
    logging.info('Summary: %d events form %d pairs'
          % (len(graph_of_connected_events.nodes()), len(graph_of_connected_events.edges())))
    logging.info('Forming %d polycronous groups' % len(list_of_polychronous_groups))


    # Plotting
    fig = plt.figure('Figure 8', figsize=(12, 12))
    fig.suptitle('Figure 7. Polychronous groups', fontsize=14, fontweight='bold')

    # plot example of a single polychronous group
    ax1 = plt.subplot(221)
    plot(graph_of_connected_events)
    plt.xlim((590,700))
    plt.ylim((0,60))
    shrink_axes(ax1, xshrink=0.01)
    label_subplot(ax1, 'A', xoffset=-0.05, yoffset=-0.01)

    # plot size distribution
    ax2 = plt.subplot(222)
    polychronous_group_size = list(len(g) for g in list_of_polychronous_groups)
    plot_loglog_fit(ax2, polychronous_group_size, rank_threshold=100)  #exclude 100 largest groups from the fit

    label_subplot(ax2, 'B', xoffset=-0.06, yoffset=-0.01)

    # plot example of a single polychronous group with arrows
    ax3 = plt.subplot(223)
    polychronous_group = list_of_polychronous_groups[3]   #10 too complex, 7 highly repetetive
    plot_pcg(ax3, polychronous_group)
    plt.ylim((0,60))
    shrink_axes(ax3, xshrink=0.01)
    label_subplot(ax3, 'C', xoffset=-0.05, yoffset=-0.01)


    # plot example of a single polychronous group onto network
    ax4 = plt.subplot(224)
    trigger, _, axon_delay, dendrite_peak = load_compartments(FIGURE_ARBORS_FILE)
    pos = load_positions(HIDENS_ELECTRODES_FILE)
    neuron_pos = neuron_position_from_trigger_electrode (pos, trigger)
    all_delay = pickle.load(open('temp/all_delays.p', 'rb'))
    plot_pcg_on_network(ax4, polychronous_group, all_delay, neuron_pos)
    ax4.set_title('network activation by polychronous group')
    label_subplot(ax4, 'D', xoffset=-0.04, yoffset=-0.01)


    # # plot polychronous groups with in different arrow colors
    # ax = plt.subplot(111)
    # plot_pcgs(ax, list_of_polychronous_groups[0:2])
    plt.show()


# testing()
figure08()
