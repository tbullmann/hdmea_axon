from hana.recording import load_positions, load_timeseries, HIDENS_ELECTRODES_FILE
from hana.segmentation import load_compartments, load_neurites, neuron_position_from_trigger_electrode
from hana.polychronous import filter, combine, group, plot, plot_pcg_on_network, plot_pcg
from hana.structure import all_overlaps
from publication.plotting import FIGURE_ARBORS_FILE, plot_loglog_fit, FIGURE_EVENTS_FILE

import os
import pickle
from matplotlib import pyplot as plt
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

def interval_timeseries (timeseries):
    first, last = [], []
    for neuron in timeseries:
        first.append (min(timeseries[neuron]))
        last.append(max(timeseries[neuron]))
    return min(first), max(last)

def partial_timeseries (timeseries, interval=0.1):

    begin, end = interval_timeseries(timeseries)

    if interval is not tuple:
        partial_begin, partial_end = (begin, (end-begin) * interval)
    else:
        partial_begin, partial_end = interval

    for neuron in timeseries:
        timeseries[neuron] = timeseries[neuron][np.logical_and(timeseries[neuron]>partial_begin,timeseries[neuron]<partial_end)]

    logging.info('Partial timeseries spanning %d~%d [s] of total %d~%d [s]' % (partial_begin, partial_end, begin, end))
    return timeseries

def figure08():
    if not os.path.isfile('temp/all_delays.p'):
        axon_delay, dendrite_peak = load_neurites (FIGURE_ARBORS_FILE)
        _, all_delays = all_overlaps(axon_delay, dendrite_peak)
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
    trigger, _, axon_delay, dendrite_peak = load_compartments(FIGURE_ARBORS_FILE)
    pos = load_positions(HIDENS_ELECTRODES_FILE)
    neuron_pos = neuron_position_from_trigger_electrode (pos, trigger)
    all_delay = pickle.load(open('temp/all_delays.p', 'rb'))
    plot_pcg_on_network(ax4, polychronous_group, all_delay, neuron_pos)
    ax4.set_title('network activation by polychronous group')


    # # plot polychronous groups with in different arrow colors
    # ax = plt.subplot(111)
    # plot_pcgs(ax, list_of_polychronous_groups[0:2])
    plt.show()


# testing()
figure08()
