import logging

import numpy as np
from matplotlib import pyplot as plt

from hana.matlab import load_traces, load_positions
from hana.plotting import set_axis_hidens
from hana.recording import electrode_neighborhoods, __segment_dendrite, __segment_axon
from publication.plotting import FIGURE_ELECTRODES_FILE, FIGURE_NEURON_FILE_FORMAT, FIGURE_NEURONS, \
    cross_hair, shrink_axes

logging.basicConfig(level=logging.DEBUG)

MAX_AXON_ELECTRODES = 5000  #TODO: Maybe over 50% of all electrodes is a good cutoff


# figure S1

def figureS1(neurons = None):

    # Load electrode coordinates
    pos = load_positions(FIGURE_ELECTRODES_FILE)
    neighbors = electrode_neighborhoods(pos)

    # If nothing specified, plot everything TODO: Something more flexible like reading actual filelist
    if not neurons: neurons = range(2, 63)

    index_plot = 0
    plotted_neurons = list()

    for neuron in neurons:
        # Load  data
        V, t, x, y, trigger, neuron = load_traces(FIGURE_NEURON_FILE_FORMAT % (neuron))
        t *= 1000  # convert to ms

        # Segment axon and dendrite
        _, _, _, _, _, _, index_AIS, _, axon = __segment_axon(t, V, neighbors)
        _, _, _, _, _, _, _, _, _, _, dendrite = __segment_dendrite(t, V, neighbors)
        number_axon_electrodes = sum(axon)
        number_dendrite_electrodes = sum(dendrite)
        logging.info('%d electrodes near axons, %d electrodes near dendrites' % (number_axon_electrodes, number_dendrite_electrodes))

        if number_dendrite_electrodes>0 and number_axon_electrodes>7 and number_axon_electrodes<MAX_AXON_ELECTRODES:

            plotted_neurons.append(int(neuron))

            # New figure every 6 plots
            if index_plot % 6 == 0:
                figure_title = 'Figure S1-%d' % (1 + (index_plot / 6))
                logging.info(figure_title)
                fig = plt.figure(figure_title, figsize=(12, 8))
                fig.suptitle(figure_title + '. Axons and dendrites', fontsize=14, fontweight='bold')

            # Get electrodes
            index_axon = np.where(axon)
            index_dendrite = np.where(dendrite)

            # Map of axon and dendrites
            ax = plt.subplot(231+(index_plot%6))
            shrink_axes(ax, yshrink=0.015)
            ax.scatter(x[index_axon], y[index_axon], c='blue', s=20, marker='.', edgecolor='None', label='axon', alpha=0.5)
            ax.scatter(x[index_dendrite], y[index_dendrite], c='red', s=20, marker='.', edgecolor='None', label='dendrite', alpha=0.5)
            cross_hair(ax, x[index_AIS], y[index_AIS], color='black')
            set_axis_hidens(ax, pos)
            ax.set_title('neuron %d' % (neuron))
            # legend_without_multiple_labels(ax)

            # Report plot and increase index_plot
            logging.info ('Plot %d on ' % (index_plot+1) + figure_title)
            index_plot += 1

        else:
            logging.info ('Nothing to plot.')

    plt.show()
    logging.info('Plotted neurons:')
    logging.info(plotted_neurons)


# figureS1()  # Plots all neurons
figureS1(FIGURE_NEURONS)
