from hana.recording import load_positions
from hana.plotting import mea_axes
from publication.plotting import show_or_savefig, cross_hair, adjust_position
from publication.data import Experiment, FIGURE_CULTURE

import os
import numpy as np
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)


def make_figure(figurename, figpath=None):

    # Load electrode coordinates and calculate neighborhood
    pos = load_positions(mea='hidens')

    x = pos.x
    y = pos.y

    all_triggers, all_AIS, all_axonal_delays, all_dendritic_return_currents = Experiment(FIGURE_CULTURE).compartments()

    index_plot = 0
    plotted_neurons = []

    list_of_six_neurons_each = [iter(all_axonal_delays.keys())] * 6

    for plot_index, six_neurons in enumerate(map(None, *list_of_six_neurons_each)):
        longfigurename = figurename + '-%d' % plot_index
        logging.info(longfigurename)
        fig = plt.figure(longfigurename, figsize=(12, 8))
        fig.suptitle(longfigurename + '. Axons and dendrites', fontsize=14, fontweight='bold')
        for neuron in six_neurons:
            if neuron is not None:
                plotted_neurons.append(int(neuron))

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


if __name__ == "__main__":

    make_figure(os.path.basename(__file__))

