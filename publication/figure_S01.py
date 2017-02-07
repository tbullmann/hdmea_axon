from hana.recording import load_positions
from hana.plotting import set_axis_hidens
from hana.segmentation import extract_and_save_compartments, load_compartments, load_neurites
from publication.plotting import cross_hair, adjust_position, FIGURE_ARBORS_FILE, FIGURE_NEURONS, FIGURE_NEURON_FILE_FORMAT

import os
import numpy as np
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)




# figure S1

def figureS1(neurons):

    # Load electrode coordinates and calculate neighborhood
    pos = load_positions(HIDENS_ELECTRODES_FILE)
    x = pos.x
    y = pos.y

    # all_triggers, all_AIS, all_axonal_delays, all_dendritic_return_currents = extract_all_compartments(neurons)
    if not os.path.isfile(FIGURE_ARBORS_FILE):
        extract_and_save_compartments(FIGURE_NEURON_FILE_FORMAT, FIGURE_ARBORS_FILE)
    all_triggers, all_AIS, all_axonal_delays, all_dendritic_return_currents = load_compartments(FIGURE_ARBORS_FILE)

    index_plot = 0
    plotted_neurons = []

    for neuron in all_axonal_delays.keys():

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
        adjust_position(ax, yshrink=0.015)
        ax.scatter(x[index_axon], y[index_axon], c='blue', s=20, marker='.', edgecolor='None', label='axon', alpha=0.5)
        ax.scatter(x[index_dendrite], y[index_dendrite], c='red', s=20, marker='.', edgecolor='None', label='dendrite',
                   alpha=0.5)
        cross_hair(ax, x[index_AIS], y[index_AIS], color='black')
        set_axis_hidens(ax)
        ax.set_title('neuron %d' % (neuron))
        # legend_without_multiple_labels(ax)

        # Report plot and increase index_plot
        logging.info('Plot %d on ' % (index_plot + 1) + figure_title)
        index_plot += 1

    plt.show()
    logging.info('Plotted neurons:')
    logging.info(plotted_neurons)


# Testing extraction

def test_neurite_extraction():
    extract_and_save_compartments(FIGURE_NEURON_FILE_FORMAT, FIGURE_ARBORS_FILE)

    trigger, AIS, delay, positive_peak = load_neurites(FIGURE_ARBORS_FILE)
    print('NEW: neuron indices as keys')
    print(trigger.keys())

    # old version (extraction was done in matlab)
    import hana
    from publication.plotting import FIGURE_ARBORS_MATFILE

    delay2, positive_peak2 = hana.matlab.load_neurites(FIGURE_ARBORS_MATFILE)
    print('OLD: electrode indices as keys')
    print (delay2.keys())


# test_neurite_extraction()
# figureS1(range(2, 63))  # Plots all neurons
figureS1(FIGURE_NEURONS)

