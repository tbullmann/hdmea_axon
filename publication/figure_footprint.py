import os
import numpy as np
from matplotlib import pyplot as plt
import logging

from hana.plotting import annotate_x_bar, mea_axes
from hana.recording import DELAY_EPSILON, neighborhood, electrode_neighborhoods
from hana.segmentation import segment_axon_verbose, restrict_to_compartment

from publication.data import FIGURE_CULTURE, FIGURE_NEURON
from publication.experiment import Experiment
from publication.plotting import show_or_savefig, without_spines_and_ticks, cross_hair, \
    legend_without_multiple_labels, plot_traces_and_delays, adjust_position

logging.basicConfig(level=logging.DEBUG)


def make_figure(figurename, figpath=None):

    # Load electrode coordinates
    neighbors = electrode_neighborhoods(mea='hidens')

    # Load example data
    V, t, x, y, trigger, neuron = Experiment(FIGURE_CULTURE).traces(FIGURE_NEURON)
    t *= 1000  # convert to ms

    # Verbose axon segmentation function
    delay, mean_delay, std_delay, expected_std_delay, thr, valid_delay, index_AIS, positive_delay, axon \
        = segment_axon_verbose(t, V, neighbors)

    logging.info ('Axonal delays:')
    logging.info (restrict_to_compartment(mean_delay, axon))

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 13))
    if not figpath:
        fig.suptitle(figurename + ' Segmentation of the axon based on negative peak at neighboring electrodes',
                 fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.05)

    # 1713   472

    background_color = 'silver'
    time_factor = 2
    voltage_factor = - 0.5
    dot_voltage = 5
    delay_factor = 1./max(delay)

    ax = plt.subplot(111)
    cmap = plt.cm.hsv
    for el in range(0,len(V)):
    #for el in range(3000, 3100):
        print('Trace at electrode %d plotted' % el)
        el_time = time_factor * t + x[el]
        el_voltage = voltage_factor * V[el] + y[el]
        el_color = cmap(delay_factor * delay[el]) if axon[el] else background_color

        # plot trace
        ax.plot(el_time, el_voltage, color=el_color, linewidth=0.5)

        # add marker for t=0
        ax.scatter(x[el], y[el] + voltage_factor * dot_voltage, 1, color=background_color, marker='.')

    cross_hair(ax, x[index_AIS], y[index_AIS])

    mea_axes(ax)
    # ax.set_xlim((180,1650))
    # ax.set_ylim((450,2100))

    show_or_savefig(figpath, figurename, dpi=600)


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))