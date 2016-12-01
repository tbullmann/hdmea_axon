import logging

import numpy as np
from matplotlib import pyplot as plt

from hana.matlab import load_traces, load_positions
from hana.plotting import set_axis_hidens
from hana.recording import electrode_neighborhoods, find_AIS, segment_dendrite, segment_axon
from publication.plotting import FIGURE_ELECTRODES_FILE, FIGURE_NEURON_FILE_FORMAT, FIGURE_NEURONS, \
    cross_hair, shrink_axes

MIN_DENDRITE_ELECTRODES = 0  #TODO: Maybe at least one electrode
MIN_AXON_ELECTRODES = 7  #TODO: Maybe at least one neighborhood
MAX_AXON_ELECTRODES = 5000  #TODO: Maybe over 50% of all electrodes is a good cutoff

logging.basicConfig(level=logging.DEBUG)




# figure S1

def figureS1(neurons):

    # Load electrode coordinates and calculate neighborhood
    pos = load_positions(FIGURE_ELECTRODES_FILE)
    x = pos.x
    y = pos.y

    all_triggers, all_AIS, all_axonal_delays, all_dendritic_return_currents = extract_all_compartments(neurons)

    index_plot = 0
    plotted_neurons = []

    for neuron in neurons:

        plotted_neurons.append(int(neuron))

        # New figure every 6 plots
        if index_plot % 6 == 0:
            figure_title = 'Figure S1-%d' % (1 + (index_plot / 6))
            logging.info(figure_title)
            fig = plt.figure(figure_title, figsize=(12, 8))
            fig.suptitle(figure_title + '. Axons and dendrites', fontsize=14, fontweight='bold')

        # Get electrodes
        index_AIS = all_AIS[neuron]
        index_axon = np.where(np.isfinite(all_axonal_delays[neuron]))
        index_dendrite = np.where(np.isfinite(all_dendritic_return_currents[neuron]))

        # Map of axon and dendrites
        ax = plt.subplot(231 + (index_plot % 6))
        shrink_axes(ax, yshrink=0.015)
        ax.scatter(x[index_axon], y[index_axon], c='blue', s=20, marker='.', edgecolor='None', label='axon', alpha=0.5)
        ax.scatter(x[index_dendrite], y[index_dendrite], c='red', s=20, marker='.', edgecolor='None', label='dendrite',
                   alpha=0.5)
        cross_hair(ax, x[index_AIS], y[index_AIS], color='black')
        set_axis_hidens(ax, pos)
        ax.set_title('neuron %d' % (neuron))
        # legend_without_multiple_labels(ax)

        # Report plot and increase index_plot
        logging.info('Plot %d on ' % (index_plot + 1) + figure_title)
        index_plot += 1

    plt.show()
    logging.info('Plotted neurons:')
    logging.info(plotted_neurons)


def extract_all_compartments(neurons):

    # Load electrode coordinates and calculate neighborhood
    pos = load_positions(FIGURE_ELECTRODES_FILE)
    neighbors = electrode_neighborhoods(pos)

    # Initialize dictionaries
    extracted_neurons = []
    all_triggers = {}
    all_AIS = {}
    all_axonal_delays = {}
    all_dendritic_return_currents = {}

    for neuron in neurons:
        # Load  data
        V, t, x, y, trigger, _ = load_traces(FIGURE_NEURON_FILE_FORMAT % (neuron))
        t *= 1000  # convert to ms

        axon, dendrite, axonal_delay, dendrite_return_current, index_AIS, number_axon_electrodes, \
        number_dendrite_electrodes = extract_compartments(t, V, neighbors)

        if number_dendrite_electrodes> MIN_DENDRITE_ELECTRODES \
                and number_axon_electrodes>MIN_AXON_ELECTRODES and number_axon_electrodes<MAX_AXON_ELECTRODES:

            extracted_neurons.append(int(neuron))
            all_triggers[neuron] = trigger
            all_AIS[neuron] = index_AIS
            all_axonal_delays[neuron] = axonal_delay
            all_dendritic_return_currents[neuron] = dendrite_return_current
        else:
            logging.info('No axonal and dendritic compartment(s).')

    plt.show()
    logging.info('Neurons with axonal and dendritic arbors:')
    logging.info(extracted_neurons)

    return all_triggers, all_AIS, all_axonal_delays, all_dendritic_return_currents


def extract_compartments(t, V, neighbors):
    # Segment axon and dendrite
    index_AIS = find_AIS(V)
    axonal_delay = segment_axon(t, V, neighbors)
    axon = np.isfinite(axonal_delay)
    dendrite_return_current = segment_dendrite(t, V, neighbors)
    dendrite = np.isfinite(dendrite_return_current)
    number_axon_electrodes = sum(axon)
    number_dendrite_electrodes = sum(dendrite)
    logging.info(
        '%d electrodes near axons, %d electrodes near dendrites' % (number_axon_electrodes, number_dendrite_electrodes))
    return axon, dendrite, axonal_delay, dendrite_return_current, index_AIS, number_axon_electrodes, number_dendrite_electrodes



# figureS1(range(2, 63))  # Plots all neurons
figureS1(FIGURE_NEURONS)


# TODO: recordings.extract_neurites (extract and save as hdf5),  recordings.load_neurites, compare structure with  matlab.load_neurites
