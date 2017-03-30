from __future__ import division

import logging
import os
import pickle
import matplotlib.pyplot as plt

from hana.misc import unique_neurons
from hana.plotting import plot_neuron_points, plot_network, mea_axes
from hana.recording import load_positions
from hana.segmentation import neuron_position_from_trigger_electrode, load_compartments
from publication.plotting import show_or_savefig, FIGURE_ARBORS_FILE, TEMPORARY_PICKELED_NETWORKS, compare_networks, \
    label_subplot, plot_synapse_delays, DataFrame_from_Dicts

logging.basicConfig(level=logging.DEBUG)

def make_figure(figurename, figpath=None):

    structural_strength, structural_delay, functional_strength, functional_delay \
        = pickle.load( open(TEMPORARY_PICKELED_NETWORKS, 'rb'))

    # Getting and subsetting the data
    data = DataFrame_from_Dicts(functional_delay, functional_strength, structural_delay, structural_strength)
    delayed = data[data.delayed]
    simultaneous = data[data.simultaneous]

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 6))
    fig.suptitle(figurename + ' Putative chemical synaptic connections', fontsize=14, fontweight='bold')

    ax1 = plt.subplot(121)
    plot_synapse_delays(ax1, data, ylim=(-2, 7))

    ax2 = plt.subplot(122)
    trigger, _, _, _ = load_compartments(FIGURE_ARBORS_FILE)
    pos = load_positions(mea='hidens')
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)
    plot_network(ax2, zip(simultaneous.pre, simultaneous.post), neuron_pos, color='red')
    plot_network(ax2, zip(delayed.pre, delayed.post), neuron_pos, color='green')
    plot_neuron_points(ax2, unique_neurons(structural_delay), neuron_pos)
    # plot_neuron_id(ax2, trigger, neuron_pos)
    # Legend by proxy
    ax2.hlines(0,0,0,linestyle='-',color='red',label='<1ms')
    ax2.hlines(0,0,0,linestyle='-',color='green',label='>1ms')
    ax2.text(200,200,r'$\mathsf{\rho=300\mu m^2}$', fontsize=14)
    plt.legend(frameon=False)
    mea_axes(ax2)

    # Label subplots
    label_subplot(ax1,'A', xoffset=-0.04, yoffset=-0.02)
    label_subplot(ax2,'B', xoffset=-0.04, yoffset=-0.02)

    show_or_savefig(figpath, figurename)


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))