import os
import pickle

from matplotlib import pyplot as plt

from hana.matlab import load_neurites, load_events, load_positions, events_to_timeseries
from hana.polychronous import filter, combine, group, plot, plot_pcg_on_network, plot_pcg
from hana.structure import all_overlaps
from publication.plotting import FIGURE_ARBORS_MATFILE, FIGURE_EVENTS_FILE, FIGURE_ELECTRODES_FILE, plot_loglog_fit

import logging
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
        axon_delay, dendrite_peak = load_neurites (FIGURE_ARBORS_MATFILE)
        _, all_delays = all_overlaps(axon_delay, dendrite_peak)
        pickle.dump(all_delays, open('temp/all_delays.p', 'wb'))

    if not os.path.isfile('temp/partial_timeseries.p'):
        events = load_events(FIGURE_EVENTS_FILE)
        events = events[0:10000] # for figures
        timeseries = events_to_timeseries(events)
        pickle.dump(timeseries, open('temp/partial_timeseries.p', 'wb'))

    if not os.path.isfile('temp/connected_events.p'):
        all_delays = pickle.load(open('temp/all_delays.p', 'rb'))
        timeseries = pickle.load(open('temp/partial_timeseries.p', 'rb'))

        connected_events = filter(timeseries, all_delays)
        pickle.dump(connected_events, open('temp/connected_events.p', 'wb'))

    connected_events = pickle.load(open('temp/connected_events.p', 'rb'))
    graph_of_connected_events = combine(connected_events)
    list_of_polychronous_groups = group(graph_of_connected_events)

    # for debugging:
    logging.info('Summary: %d events form %d pairs'
          % (len(graph_of_connected_events.nodes()), len(graph_of_connected_events.edges())))
    logging.info('Forming %d polycronous groups' % len(list_of_polychronous_groups))


    # Plotting
    plt.figure('Figure 8', figsize=(12,12))


    # plot example of a single polychronous group
    plt.subplot(221)
    plot(graph_of_connected_events)
    plt.xlim((590,700))

    # plot size distribution
    ax2 = plt.subplot(222)
    polychronous_group_size = list(len(g) for g in list_of_polychronous_groups)
    plot_loglog_fit(ax2, polychronous_group_size)

    # plot example of a single polychronous group with arrows
    ax = plt.subplot(223)
    polychronous_group = list_of_polychronous_groups[3]   #10 too complex, 7 highly repetetive
    plot_pcg(ax, polychronous_group)

    # plot example of a single polychronous group onto network
    ax4 = plt.subplot(224)
    pos = load_positions(FIGURE_ELECTRODES_FILE)
    all_delay = pickle.load(open('temp/all_delays.p', 'rb'))
    plot_pcg_on_network(ax4, polychronous_group, all_delay, pos)
    ax4.set_title('network activation by polychronous group')


    # # plot polychronous groups with in different arrow colors
    # ax = plt.subplot(111)
    # plot_pcgs(ax, list_of_polychronous_groups[0:2])
    plt.show()


# testing()
figure08()
