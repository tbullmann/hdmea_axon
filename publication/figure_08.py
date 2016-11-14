import pickle
import os

from hana.matlab import load_neurites, load_events, events_to_timeseries
from hana.structure import all_overlaps
from hana.polychronous import filter, combine, plot, group

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
    if not os.path.isfile('temp/data_PCGs_hidens2018.p'):
        axon_delay, dendrite_peak = load_neurites ('data/hidens2018at35C_arbors.mat')
        _, all_delays = all_overlaps(axon_delay, dendrite_peak)
        events = load_events('data/hidens2018at35C_events.mat')
        timeseries = events_to_timeseries(events[1:10000])
        pickle.dump((timeseries, all_delays), open('temp/data_PCGs_hidens2018.p', 'wb'))

    timeseries, all_delays = pickle.load(open('temp/data_PCGs_hidens2018.p', 'rb'))

    connected_events = filter (timeseries, all_delays)
    graph_of_connected_events = combine (connected_events)
    list_of_polychronous_groups = group(graph_of_connected_events)

    # for debugging:
    print(graph_of_connected_events.nodes())
    print(graph_of_connected_events.edges())
    print(len(list_of_polychronous_groups))

    # plot all polychronous grooups
    plot(graph_of_connected_events)
    plt.show()

    # plot example of a single polychronous group
    polychronous_group = list_of_polychronous_groups[1]
    plot(polychronous_group)
    plt.show()


# testing()
figure08()
