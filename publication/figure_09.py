from __future__ import division

import logging
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from matplotlib.ticker import NullFormatter

from hana.misc import unique_neurons
from hana.recording import load_positions
from hana.segmentation import neuron_position_from_trigger_electrode, load_compartments
from hana.plotting import plot_neuron_points, plot_neuron_id, plot_network, set_axis_hidens
from publication.plotting import FIGURE_ARBORS_FILE, TEMPORARY_PICKELED_NETWORKS, compare_networks, \
    label_subplot, correlate_two_dicts_verbose, kernel_density, axes_to_3_axes

logging.basicConfig(level=logging.DEBUG)

def figure09 (networks_pickel_name):

    structural_strengths, structural_delays, functional_strengths, functional_delays = pickle.load( open(networks_pickel_name, 'rb'))

    # Making figure
    fig = plt.figure('Figure 9', figsize=(13, 6))
    fig.suptitle('Figure 9. Putative chemical synaptic connections', fontsize=14, fontweight='bold')

    structural_thresholds, functional_thresholds, intersection_size, structural_index, jaccard_index, functional_index\
        = compare_networks(structural_strengths, functional_strengths, scale='log')

    ax1 = plt.subplot(121)
    delayed_pairs, simultaneous_pairs, synpase_delays = plot_synapse_delays(ax1, structural_delays, functional_delays, functional_strengths, ylim=(-2,7))

    ax2 = plt.subplot(122)
    trigger, _, _, _ = load_compartments(FIGURE_ARBORS_FILE)
    pos = load_positions(mea='hidens')
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)
    plot_neuron_points(ax2, unique_neurons(structural_delays), neuron_pos)
    plot_neuron_id(ax2, trigger, neuron_pos)
    # plot_network(ax2, structural_delays, neuron_pos, color='gray')
    plot_network(ax2, simultaneous_pairs, neuron_pos, color='red')
    plot_network(ax2, delayed_pairs, neuron_pos, color='green')
    # Legend by proxy
    ax2.hlines(0,0,0,linestyle='-',color='red',label='<1ms')
    ax2.hlines(0,0,0,linestyle='-',color='green',label='>1ms')
    ax2.text(200,200,r'$\mathsf{\rho=300\mu m^2}$', fontsize=14)
    plt.legend(frameon=False)
    set_axis_hidens(ax2)

    # Label subplots
    label_subplot(ax1,'A', xoffset=-0.04, yoffset=-0.02)
    label_subplot(ax2,'B', xoffset=-0.04, yoffset=-0.02)

    plt.show()


def plot_synapse_delays(ax, structural_delay, functional_delay, functional_strength, xlim=None, ylim=None, xscaling='log', yscaling ='count', naxes=3):
    """Plot corretion and marginals"""
    # New axes
    if naxes==2:
        axScatter=ax
        fig = ax.get_figure()
        divider = make_axes_locatable(axScatter)
        axHisty = divider.new_horizontal(size="50%", pad=0.05)
        fig.add_axes(axHisty)
    if naxes==3:
        rect_histx, rect_histy, rect_scatter = axes_to_3_axes(ax)
        axScatter = plt.axes(rect_scatter)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)

    # getting the data
    delay_axon, timing_spike, pairs = correlate_two_dicts_verbose(structural_delay, functional_delay)
    # delay_synapse = timing_spike - delay_axon
    # __, strength_synapse, __ = __correlate_two_dicts(structural_delay, functional_strength)
    synapse_delay = dict(zip(pairs, timing_spike - delay_axon))   # create a new dictionary of synaptic delays
    tau_synapse, z_max, pairs = correlate_two_dicts_verbose(synapse_delay, functional_strength)

    # Find putative chemical synapse with synaptic delay > 1ms, and other with delays <= 1ms
    delayed_indices = np.where(tau_synapse>1)
    delayed_pairs = np.array(pairs)[delayed_indices]
    simultaneous_indices = np.where(tau_synapse<=1)
    simultaneous_pairs = np.array(pairs)[simultaneous_indices]

    # scatter plot
    axScatter.scatter(z_max, tau_synapse, color='red', label='<1ms')
    axScatter.scatter(z_max[delayed_indices],tau_synapse[delayed_indices], color='green', label='>1ms')
    axScatter.set_xscale(xscaling)
    axScatter.legend(frameon=False, scatterpoints=1)
    axScatter.set_xlabel(r'$\mathsf{z_{max}}$', fontsize=14)
    axScatter.set_ylabel(r'$\mathsf{\tau_{synapse}\ [ms]}$', fontsize=14)

    # density plot
    kernel_density(axHisty, tau_synapse, scaling=yscaling, style='k-', orientation='horizontal')
    if naxes==3:
        # joint legend by proxies
        plt.sca(ax)
        plt.vlines(0, 0, 0, colors='green', linestyles='-', label='>1ms')
        plt.vlines(0, 0, 0, colors='red', linestyles='-', label='<1ms')
        plt.vlines(0, 0, 0, colors='black', linestyles='-', label='all')
        plt.legend(frameon=False, fontsize=12)

        # kernel_density(axHistx, strength_synapse, scaling=yscaling, style='k-', orientation='vertical')
        kernel_density(axHistx, z_max[delayed_indices], scaling=yscaling, style='g-', orientation='vertical')
        kernel_density(axHistx, z_max[simultaneous_indices], scaling=yscaling, style='r-', orientation='vertical')
        axHistx.set_xscale(xscaling)

    # define limits
    max_strength_synapse = max(z_max)
    if not xlim: xlim = (1, max_strength_synapse*2)
    if not ylim: ylim = (min(tau_synapse), max(tau_synapse))

    # set limits
    axScatter.set_xlim(xlim)
    axScatter.set_ylim(ylim)
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    # add hlines to Scatter
    axScatter.hlines(0, 0, max_strength_synapse*2, linestyles='--')
    axScatter.hlines(-1, 0, max_strength_synapse*2, linestyles=':')
    axScatter.hlines(+1, 0, max_strength_synapse*2, linestyles=':')

    # no labels
    nullfmt = NullFormatter()  # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHistx.yaxis.set_major_formatter(nullfmt)
    axHisty.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    return delayed_pairs, simultaneous_pairs, synapse_delay


def dict2array_like_index(dictionary, keys):
    """
    Returns an array with elements from the dictionary preserving the order of keys. Note: Not used
    :param dictionary:
    :param keys:
    :return:
    """
    strength_synapse = []
    for key in keys:
        value = dictionary[key]
        strength_synapse.append(value)
    strength_synapse = np.array(strength_synapse)
    return strength_synapse


figure09(TEMPORARY_PICKELED_NETWORKS)