from __future__ import division

import logging
import os
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from matplotlib.ticker import NullFormatter
from scipy.stats import gaussian_kde

from hana.misc import unique_neurons
from hana.recording import load_positions
from hana.segmentation import HIDENS_ELECTRODES_FILE, neuron_position_from_trigger_electrode, load_compartments
from hana.plotting import plot_neuron_points, plot_neuron_id, plot_network, set_axis_hidens
from publication.plotting import FIGURE_ARBORS_FILE, TEMPORARY_PICKELED_NETWORKS, compare_networks, \
    label_subplot, __correlate_two_dicts, kernel_density

logging.basicConfig(level=logging.DEBUG)

def figure09 (networks_pickel_name):

    structural_strengths, structural_delays, functional_strengths, functional_delays = pickle.load( open(networks_pickel_name, 'rb'))

    # Making figure
    fig = plt.figure('Figure 9', figsize=(16, 8))
    fig.suptitle('Figure 9. Putative chemical synaptic connections', fontsize=14, fontweight='bold')

    structural_thresholds, functional_thresholds, intersection_size, structural_index, jaccard_index, functional_index\
        = compare_networks(structural_strengths, functional_strengths, scale='log')

    ax1 = plt.subplot(121)
    label_subplot(ax1,'A', xoffset=-0.04,yoffset=-0.02)
    pairs = plot_synapse_delays(ax1, structural_delays, functional_delays, xlim=(0,4), ylim=(-2,5))
    ax1.set_xlabel ('axonal delays [ms]')
    ax1.set_ylabel ('spike timing [s]')

    ax2 = plt.subplot(122)
    trigger, _, _, _ = load_compartments(FIGURE_ARBORS_FILE)
    pos = load_positions(HIDENS_ELECTRODES_FILE)
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)
    plot_neuron_points(ax2, unique_neurons(structural_delays), neuron_pos)
    plot_neuron_id(ax2, trigger, neuron_pos)
    plot_network(ax2, structural_delays, neuron_pos, color='gray')
    plot_network(ax2, pairs, neuron_pos, color='black')
    ax2.hlines(0,0,0,linestyle='-',color='gray',label='all')
    ax2.hlines(0,0,0,linestyle='-',color='black',label='putative chemical')
    plt.legend(frameon=False)

    set_axis_hidens(ax2)
    label_subplot(ax2,'B',xoffset=-0.04,yoffset=-0.02)

    plt.show()


def plot_synapse_delays(axScatter, xdict, ydict, xlim=None, ylim=None, scaling = 'count'):
    """Plot corretion and marginals"""
    # getting the data
    delay_axon, timing_spike, pairs = __correlate_two_dicts(xdict, ydict)
    delay_synapse = timing_spike - delay_axon

    # Find putative chemical synapse with synaptic delay > 1ms
    indices = np.where(delay_synapse>1)
    synaptic_pairs = np.array(pairs)[indices]

    # scatter plot
    axScatter.scatter(delay_axon, delay_synapse, color='gray', label='all')
    axScatter.scatter(delay_axon[indices],delay_synapse[indices],color='black', label='putative chemical')
    plt.legend(frameon=False)

    # density plot
    divider = make_axes_locatable(axScatter)
    axHisty = divider.new_horizontal(size="50%", pad=0.05)
    fig = axScatter.get_figure()
    fig.add_axes(axHisty)
    kernel_density(axHisty, delay_synapse, scaling=scaling, style='k-', orientation='horizontal')

    # define limits
    if not xlim: xlim = (min(delay_axon), max(delay_axon))
    if not ylim: ylim = (min(delay_synapse), max(delay_synapse))

    # set limits
    axScatter.set_xlim(xlim)
    axScatter.set_ylim(ylim)
    axHisty.set_ylim(axScatter.get_ylim())

    # add hlines to Scatter
    axScatter.hlines(0,0,4,linestyles='--')
    axScatter.hlines(-1,0,4,linestyles=':')
    axScatter.hlines(+1,0,4,linestyles=':')


    # no labels
    nullfmt = NullFormatter()  # no labels

    axHisty.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    return synaptic_pairs

figure09(TEMPORARY_PICKELED_NETWORKS)