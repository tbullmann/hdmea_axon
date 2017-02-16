from hana.misc import unique_neurons
from hana.plotting import plot_neuron_points, plot_network, highlight_connection, mea_axes

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

import logging
logging.basicConfig(level=logging.DEBUG)


def filter(timeseries, axonal_delays, synaptic_delay=0.001, jitter=0.001):
    """
    Shift presynaptic spike by timelag predicted from axonal and synaptic delay. Shifted presynaptic spikes and
    post synaptic spikes that match timing within a jitter form pairs of pre- and post-synaptic events, which could
    be the result of a synaptic transmission. See Izhekevich, 2006 for further explanation.
    :param timeseries: dict of neuron_id: vector of time
    :param axonal_delays: dict of (pre_neuron_id, post_neuron_id): axonal_delay in ms(!)
    :param synaptic_delay: single value, in s(!)
    :param jitter: single value, representing maximum allowed synaptic jitter (+/-), in s(!)
    :return: connected_events: tupels (time, pre_neuron_id), (time, post_neuron_id)
    """
    # TODO: Improve function description, remove unit inconsistencies (ms vs. s)
    connected_events = []
    for pre, post in axonal_delays:
        time_lag = (axonal_delays[pre, post])/1000 + synaptic_delay  # axonal delays in ms -> events in s
        logging.info("Finding spike pairs %d -> %d with predicted spike time lag %f s:" % (pre, post, time_lag))

        if (pre in timeseries) and (post in timeseries):
            presynaptic_spikes = timeseries[pre]
            shifted_presynaptic_spikes = presynaptic_spikes+time_lag
            postsynaptic_spikes = timeseries[post]
            for offset in (0, 1):  # checking postsynaptic spike before and after shifted presynaptic spike
                position = (np.searchsorted(postsynaptic_spikes, shifted_presynaptic_spikes) - offset).\
                    clip(0, len(postsynaptic_spikes) - 1)
                valid = np.abs(shifted_presynaptic_spikes - postsynaptic_spikes[position]) < jitter
                valid_presynaptic_spikes = presynaptic_spikes[valid]
                valid_postsynaptic_spikes = postsynaptic_spikes[position[valid]]
                new_connected_events = [((pre_time, pre), (post_time, post))
                                        for pre_time, post_time
                                        in (zip(valid_presynaptic_spikes, valid_postsynaptic_spikes))]
                logging.info("From %d candidates add %d valid to %d existing pairs"
                             % (len(position), len(new_connected_events), len(connected_events)))
                connected_events = connected_events + new_connected_events

    logging.info("Total %d pairs" % len(connected_events))
    return connected_events


def combine(connected_events):
    """
    Combine connected events into a graph.
    :param connected_events: see polychronous.filter
    :return: graph_of_connected_events
    """
    graph_of_connected_events = nx.Graph()
    graph_of_connected_events.add_edges_from(connected_events)
    return (graph_of_connected_events)


def plot(graph_of_connected_events):
    """
    Plot polychronous group(s) from the graph of connected events. The events are plotted at their time vs. neuron_id.
    :param graph_of_connected_events: see polychronous.combine
    """
    nx.draw_networkx(graph_of_connected_events,
                     pos={event: event for event in graph_of_connected_events.nodes()},
                     with_labels=False,
                     node_size=20,
                     node_color='black',
                     edge_color='red')
    plt.xlabel("time [s]")
    plt.ylabel("neuron index")


def group(graph_of_connected_events):
    """
    Split the graph into its connected components, each representing a polychronous group.
    :param graph_of_connected_events: see polychronous.combine
    :return: list_of_polychronous_groups
    """
    list_of_polychronous_groups = list(nx.connected_component_subgraphs(graph_of_connected_events))
    return list_of_polychronous_groups


def shuffle_keys(dictionary):
    """
    Preparing surrogate data for polychronous.filter.
    Note: Does not change the network edge connectivity distribution.
    :param dictionary: could be either a time series or delays dictionary
    :return: dictionary with the shuffled correspondence of keys to values
    """
    keys = dictionary.keys()
    shuffled_keys = random.shuffle(keys)
    new_dictionary = dict(zip(shuffled_keys, dictionary.values()))
    return new_dictionary


def plot_pcg(ax, polychronous_group, color='r'):
    """
    Plots a polychronous group with colored arrows. Unfortunately, (re)plotting is slow for large groups.
    :param ax: axis handle
    :param polychronous_group: graph containing the events and connections of a polychronous group
    :param color: color of arrows, default is red
    :return:
    """
    times, neurons = (zip(*polychronous_group.nodes()))
    ax.plot(times, neurons, '.k', markersize=10)
    for ((time1, neuron1), (time2, neuron2)) in polychronous_group.edges():
        if time2 < time1: time1, neuron1, time2, neuron2 = time2, neuron2, time1, neuron1
        ax.annotate('', (time1, neuron1), (time2, neuron2), arrowprops={'color': color, 'arrowstyle': '<-'})
    # Annotations coordinates are opposite to what expected!
    # Annotations alone do not resize the plotting windows, do:
    # ax.set_xlim([min(times), max(times)])
    # ax.set_ylim([min(neurons), max(neurons)])
    plt.xlabel("time [s]")
    plt.ylabel("neuron index")

def plot_pcg_on_network(ax, polychronous_group, delays, pos):
    """
    Plots a polychronous group onto a structural network. Unfortunately, (re)plotting is slow for large groups.
    :param ax: axis handle
    :param polychronous_group: graph containing the events and connections of a polychronous group
    :param delays: delays between neurons (structural network)
    :param pos: positions of the neurons on the multi electrode array
    """
    plot_neuron_points(ax, unique_neurons(delays), pos)
    plot_network(ax, delays, pos, color='gray')
    for i, ((time1, neuron1), (time2, neuron2)) in enumerate(polychronous_group.edges(), start=1):
        if time2 < time1: time1, neuron1, time2, neuron2 = time2, neuron2, time1, neuron1
        highlight_connection(ax, (neuron1, neuron2), pos)
    mea_axes(ax)


def plot_pcgs(ax, list_of_polychronous_groups):
    """
    Plots each polychronous group with different (arrow) colors. (Re)Plotting is painfully slow.
    :param ax: axis handle
    :param list_of_polychronous_groups: list of polychronous groups
    """
    from itertools import cycle
    cycol = cycle('bgrcmk').next
    for g in list_of_polychronous_groups:
        plot_pcg(ax, g, color=cycol())