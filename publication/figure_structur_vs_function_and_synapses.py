from __future__ import division

import logging
import os
import pickle

import matplotlib.pyplot as plt

from hana.misc import unique_neurons
from hana.plotting import plot_network, plot_neuron_points, mea_axes
from hana.recording import average_electrode_area
from hana.segmentation import load_compartments, load_positions, neuron_position_from_trigger_electrode
from publication.figure_structur_vs_function import maybe_get_functional_and_structural_networks, plot_vs_weigth
from publication.plotting import show_or_savefig, FIGURE_ARBORS_FILE, FIGURE_EVENTS_FILE, TEMPORARY_PICKELED_NETWORKS, \
    plot_correlation, plot_synapse_delays, adjust_position, DataFrame_from_Dicts

logging.basicConfig(level=logging.DEBUG)

def make_figure(figurename, figpath=None):

    maybe_get_functional_and_structural_networks(FIGURE_ARBORS_FILE, FIGURE_EVENTS_FILE, TEMPORARY_PICKELED_NETWORKS)

    structural_strength, structural_delay, functional_strength, functional_delay \
        = pickle.load( open(TEMPORARY_PICKELED_NETWORKS, 'rb'))

    # Getting and subsetting the data
    data = DataFrame_from_Dicts(functional_delay, functional_strength, structural_delay, structural_strength)
    delayed = data[data.delayed]
    simultaneous = data[data.simultaneous]

    # Map electrode number to area covered by that electrodes
    electrode_area = average_electrode_area(None, mea='hidens')
    structural_strength = {key: electrode_area*value for key, value in structural_strength.items()}

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 13))
    if not figpath:
        fig.suptitle(figurename + ' Compare structural and functional connectivity', fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.05)

    # plot network measures
    ax1 = plt.subplot(4,2,1)
    plot_vs_weigth(ax1, structural_strength)
    ax1.set_xlabel(r'$\mathsf{\rho\ [\mu m^2]}$', fontsize=14)
    xlim = ax1.get_xlim()
    ax1.text(xlim[0], 230, '                       structural connectivity', fontsize=14, horizontalalignment='left', verticalalignment='top')
    plt.title('a', loc='left', fontsize=18)

    ax2 = plt.subplot(4,2,3)
    plot_vs_weigth(ax2, functional_strength)
    ax2.set_xlabel(r'$\mathsf{\zeta}$', fontsize=14)
    xlim = ax2.get_xlim()
    ax2.text(xlim[0], 550, '                       functional connectivity', fontsize=14, horizontalalignment='left', verticalalignment='top')
    plt.title('b', loc='left', fontsize=18)

    ax4 = plt.subplot(223)
    plot_synapse_delays(ax4, data, ylim=(-2, 7))
    plt.title('d', loc='left', fontsize=18)

    ax3 = plt.subplot(2,2,2)
    adjust_position(ax3, xshift=0.04, yshift=-0.01)
    axScatter1 = plot_correlation(ax3, data, x='structural_strength', y='functional_strength', xscale='log', yscale='log')
    axScatter1.set_xlabel(r'$\mathsf{|A \cap D|\ [\mu m^2}$]', fontsize=14)
    axScatter1.set_ylabel(r'$\mathsf{z_{max}}$', fontsize=14)
    plt.title('c', loc='left', fontsize=18)

    ax5 = plt.subplot(224)
    trigger, _, _, _ = load_compartments(FIGURE_ARBORS_FILE)
    pos = load_positions(mea='hidens')
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)
    plot_network(ax5, zip(simultaneous.pre, simultaneous.post), neuron_pos, color='red')
    plot_network(ax5, zip(delayed.pre, delayed.post), neuron_pos, color='green')
    plot_neuron_points(ax5, unique_neurons(structural_delay), neuron_pos)
    # plot_neuron_id(ax2, trigger, neuron_pos)
    # Legend by proxy
    ax5.hlines(0, 0, 0, linestyle='-', color='red', label='<1ms')
    ax5.hlines(0, 0, 0, linestyle='-', color='green', label='>1ms')
    ax5.text(200, 250, r'$\mathsf{\rho=300\mu m^2}$', fontsize=18)
    ax5.text(200, 350, r'$\mathsf{\zeta=1}$', fontsize=18)
    plt.legend(frameon=False)
    mea_axes(ax5)
    adjust_position(ax5, yshift=-0.01)
    plt.title('e     synaptic delay graph', loc='left', fontsize=18)


    show_or_savefig(figpath, figurename)


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))