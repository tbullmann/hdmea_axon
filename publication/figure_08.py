import pickle
import os

from hana.matlab import load_neurites, load_events, events_to_timeseries
from hana.structure import all_overlaps
from hana.polychronous import filter, combine, plot, group
from publication.plotting import FIGURE_ARBORS_FILE, FIGURE_EVENTS_FILE

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

def figure08():
    if not os.path.isfile('temp/all_delays.p'):
        axon_delay, dendrite_peak = load_neurites (FIGURE_ARBORS_FILE)
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

    # plot example of a single polychronous group
    polychronous_group = list_of_polychronous_groups[10]
    plot(polychronous_group)
    plt.show()

    # plot all polychronous grooups
    plot(graph_of_connected_events)
    plt.show()


# testing()
figure08()
