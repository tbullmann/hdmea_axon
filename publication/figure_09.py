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
    label_subplot, correlate_two_dicts, kernel_density

logging.basicConfig(level=logging.DEBUG)

def figure09 (networks_pickel_name):

    structural_strengths, structural_delays, functional_strengths, functional_delays = pickle.load( open(networks_pickel_name, 'rb'))

    # Making figure
    fig = plt.figure('Figure 8', figsize=(16, 9))
    fig.suptitle('Figure 8. Compare structural and functional connectivity', fontsize=14, fontweight='bold')

    structural_thresholds, functional_thresholds, intersection_size, structural_index, jaccard_index, functional_index\
        = compare_networks(structural_strengths, functional_strengths, scale='log')

    ax1 = plt.subplot(121)
    label_subplot(ax1,'A')
    axScatter2 = plot_synapse_delays(ax1, structural_delays, functional_delays, xlim=(0,4), ylim=(-2,5))
    ax1.set_xlabel ('axonal delays [ms]')
    ax1.set_ylabel ('spike timing [s]')

    ax2 = plt.subplot(122)
    trigger, _, _, _ = load_compartments(FIGURE_ARBORS_FILE)
    pos = load_positions(HIDENS_ELECTRODES_FILE)
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)
    plot_neuron_points(ax2, unique_neurons(structural_delays), neuron_pos)
    plot_neuron_id(ax2, trigger, neuron_pos)
    plot_network(ax2, structural_delays, neuron_pos)
    set_axis_hidens(ax2)
    label_subplot(ax1,'B')

    plt.show()


def plot_synapse_delays(axScatter, xdict, ydict, xlim=None, ylim=None, scaling = 'count'):
    """Plot corretion and marginals"""
    # getting the data
    x_all = xdict.values()
    y_all = ydict.values()
    x_corr, y_corr = correlate_two_dicts(xdict, ydict)
    y_corr = y_corr - x_corr

    # new axes
    divider = make_axes_locatable(axScatter)
    axHisty = divider.new_horizontal(size="50%", pad=0.05)
    fig = axScatter.get_figure()
    fig.add_axes(axHisty)

    # two plots:
    axScatter.scatter(x_corr, y_corr, color='gray')
    kernel_density(axHisty, y_corr, scaling=scaling, style='k-', orientation='horizontal')

    #TODO Fit 3 Gaussians


    # define limits
    if not xlim: xlim = (min(x_corr), max(x_corr))
    if not ylim: ylim = (min(y_corr), max(y_corr))

    # set limits
    axScatter.set_xlim(xlim)
    axScatter.set_ylim(ylim)
    axHisty.set_ylim(axScatter.get_ylim())

    # add hlines to Scatter



    # no labels
    nullfmt = NullFormatter()  # no labels

    axHisty.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

figure09(TEMPORARY_PICKELED_NETWORKS)