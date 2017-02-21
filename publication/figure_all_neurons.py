from hana.recording import load_positions
from hana.plotting import mea_axes
from hana.segmentation import extract_and_save_compartments, load_compartments, load_neurites
from publication.plotting import show_or_savefig, cross_hair, adjust_position, FIGURE_ARBORS_FILE, FIGURE_NEURONS, FIGURE_NEURON_FILE_FORMAT

import os
import numpy as np
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)




# figure S1

def make_figure(figurename, figpath=None, neurons=FIGURE_NEURONS):

    # Load electrode coordinates and calculate neighborhood
    pos = load_positions(mea='hidens')

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
            longfigurename = figurename +'-%d' % (1 + (index_plot / 6))
            logging.info(longfigurename)
            fig = plt.figure(longfigurename, figsize=(12, 8))
            fig.suptitle(longfigurename + '. Axons and dendrites', fontsize=14, fontweight='bold')

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
        mea_axes(ax)
        ax.set_title('neuron %d' % (neuron))
        # legend_without_multiple_labels(ax)

        # Report plot and increase index_plot
        logging.info('Plot %d on ' % (index_plot + 1) + longfigurename)
        index_plot += 1

        show_or_savefig(figpath, longfigurename)

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

if __name__ == "__main__":
    make_figure()
    # test_neurite_extraction()
    # figureS1(range(2, 63))  # Plots all neurons
    make_figure(os.path.basename(__file__))
