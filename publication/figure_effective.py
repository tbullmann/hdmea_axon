from __future__ import division

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hana.misc import unique_neurons
from hana.plotting import plot_network, plot_neuron_points, mea_axes, plot_neuron_id
from hana.recording import average_electrode_area
from hana.segmentation import load_positions, neuron_position_from_trigger_electrode
from publication.comparison import print_test_result
from publication.data import Experiment, FIGURE_CULTURE, FIGURE_CULTURES
from publication.plotting import show_or_savefig, adjust_position, \
    without_spines_and_ticks

logging.basicConfig(level=logging.DEBUG)


def make_figure(figurename, figpath=None):

    structural_strength, structural_delay, functional_strength, synaptic_delay, synapse_strength, synapse_delay \
        = Experiment(FIGURE_CULTURE).networks()

    timelags, std_score_dict, timeseries_hist_dict = Experiment(FIGURE_CULTURE).standardscores()
    minimal_synaptic_delay = 1.0

    # Map electrode number to area covered by that electrodes
    electrode_area = average_electrode_area(None, mea='hidens')

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 13))
    if not figpath:
        fig.suptitle(figurename + ' Constrained functional connectivity', fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.05)

    remaining_keys = [key for key in structural_delay.keys() if not key in synapse_delay.keys()]

    pos = load_positions(mea='hidens')

    # Functional vs structural delays
    ax1 = plt.subplot2grid((2,4), (0,0), colspan=1, rowspan=1)

    _, structural_delay, _, functional_delay, _, _ = Experiment(FIGURE_CULTURE).networks()
    x = list()
    y = list()
    for pair in structural_delay:
        if pair in functional_delay:
            x.append(structural_delay[pair])
            y.append(functional_delay[pair])

    ax1.scatter(x, y, color='black')
    ax1.plot([0,3],[0,3],color='blue')
    ax1.fill([0,3,3,0], [0,3,0,0], fill=False, hatch='\\', linewidth=0)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 6)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel(r'$\mathsf{\tau_{axon}\ [ms]}$', fontsize=14)
    ax1.set_ylabel(r'$\mathsf{\tau_{spike}\ [ms]}$', fontsize=14)
    ax1.set_title ('culture %d' % FIGURE_CULTURE)
    adjust_position(ax1, yshrink=0.02)
    without_spines_and_ticks(ax1)
    plt.title('a', loc='left', fontsize=18)

    # Summary
    synaptic_delays = list()
    for culture in FIGURE_CULTURES:
        _, structural_delay, _, functional_delay, _, _ = Experiment(culture).networks()
        differences = list()
        for pair in structural_delay:
            if pair in functional_delay:
                differences.append(functional_delay[pair]-structural_delay[pair])
        synaptic_delays.append(differences)

    ax2 = plt.subplot2grid((2,4), (0,1), colspan=1, rowspan=1)
    plot_delays(ax2, -4, synaptic_delays, xticks = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    ax2.set_xlabel(r'$\mathsf{\tau_{spike}-\tau_{axon}\ [ms]}$', fontsize=14)
    adjust_position(ax2, yshrink=0.02, xshift=0.02)
    plt.title('b', loc='left', fontsize=18)

    # Schema
    ax3 = plt.subplot2grid((2,4), (0,2), colspan=1, rowspan=1)
    import matplotlib.image as mpimg
    img = mpimg.imread('data/sketch_effective.png')
    plt.imshow(img)
    plt.axis('off')
    plt.title('c', loc='left', fontsize=18)
    ax3.set_anchor('W')
    adjust_position(ax3, yshift=-0.005, xshift=0.02)

    # make summaries
    data = []
    synapse_delays = list()
    for culture in FIGURE_CULTURES:
        structural_strength, structural_delay, _, _, synapse_strength, synapse_delay \
            = Experiment(culture).networks()
        culture_data = DataFrame_from_Dicts(structural_delay, structural_strength, synapse_delay)
        culture_data['culture'] = culture
        data.append(culture_data)
        synapse_delays.append(synapse_delay.values())
    data = pd.concat(data)

    connections = pd.pivot_table(data, index='culture', columns='delayed',
                                 values='synaptic_delay', aggfunc = lambda x: len(x))
    connections.fillna(0, inplace=True)
    connections['delayed'] = 100* connections[True]/(connections[True] + connections[False])
    connections['simultaneous'] = 100* connections[False] / (connections[True] + connections[False])

    # Summary
    ax4 = plt.subplot2grid((2,4), (0,3), colspan=1, rowspan=1)

    print_test_result('percent of structural connections for delayed vs simultaneous', connections[['delayed', 'simultaneous']])
    bplot = ax4.boxplot(np.array(connections[['delayed','simultaneous']]),
                        boxprops={'color': "black", "linestyle": "-"}, patch_artist=True,widths=0.7)
    prettify_boxplot(ax4, bplot)
    ax4.set_ylim((0,100))
    ax4.set_ylabel('percent of structural connections [%]')
    plt.title('d', loc='left', fontsize=18)

    synaptic_delays = list()
    for culture in FIGURE_CULTURES:
        _, structural_delay, _, _, _, synaptic_delay = Experiment(culture).networks()
        effective_delay = list()
        for pair in synaptic_delay:
            effective_delay.append(structural_delay[pair] + synaptic_delay[pair])
        synaptic_delays.append(effective_delay)

    ax5 = plt.subplot2grid((4, 4), (2, 0), colspan=1, rowspan=2)
    plot_delays(ax5, minimal_synaptic_delay, synaptic_delays, fill_color='green', xticks=(0,1,2,3,4,5,6,7,8,9,10))
    ax5.set_xlabel(r'$\mathsf{\tau_{effective}=\tau_{axon}+\tau_{constrained}\ [ms]}$', fontsize=14)
    adjust_position(ax5, yshrink=0.02)
    plt.title('e', loc='left', fontsize=18)

    # all graphs
    plot_culture_synapse_graph(1, (2, 1), pos)
    plt.title('f', loc='left', fontsize=18)
    plot_culture_synapse_graph(2, (2, 2), pos)
    plot_culture_synapse_graph(3, (2, 3), pos)
    plot_culture_synapse_graph(4, (3, 1), pos)
    plot_culture_synapse_graph(5, (3, 2), pos)
    # plot_culture_synapse_graph(6, (3, 3), pos)

    show_or_savefig(figpath, figurename)


def plot_culture_synapse_graph(culture, grid_pos, pos):
    trigger, _, axon_delay, dendrite_peak = Experiment(culture=culture).compartments()
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)
    _, _, _, _, _, synaptic_delay = Experiment(culture).networks()
    neuron_dict = unique_neurons(synaptic_delay)
    axc1 = plt.subplot2grid((4, 4), grid_pos, colspan=1, rowspan=1)
    plot_network(axc1, synaptic_delay, neuron_pos, color='green')
    plot_neuron_points(axc1, neuron_dict, neuron_pos)
    mea_axes(axc1)
    axc1.set_title ('culture %d' % culture)
    adjust_position(axc1, yshrink=0.02)


def plot_delays(ax, minimal_synaptic_delay, delays, fill_color='gray', xticks = [0, 1, 2, 3, 4, 5]):
    # show distirbution of median synapse delays as boxplot
    bplot = plt.boxplot([np.median(x) for x in delays], 0, '+', 0, positions=np.array([0.]),
                        boxprops={'color': "black", "linestyle": "-"}, patch_artist=True, widths=0.7)
    bplot['boxes'][0].set_color(fill_color)
    plt.setp(bplot['medians'], color='black')
    plt.setp(bplot['whiskers'], color='black')
    # add violin plot for every culture
    vplot = plt.violinplot(delays, FIGURE_CULTURES, points=80, vert=False, widths=0.7,
                           showmeans=False, showextrema=True, showmedians=True)
    ax.fill([0, minimal_synaptic_delay, minimal_synaptic_delay, 0], [-1, -1, 9, 9],
            fill=False, hatch='\\', linewidth=0)
    # Make all the violin statistics marks black:
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = vplot[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
    # Make the violin body blue with a black border:
    for pc in vplot['bodies']:
        pc.set_facecolor(fill_color)
        pc.set_alpha(0.5)
        pc.set_edgecolor('none')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_ylabel('culture')
    ax.set_yticks([0, ] + list(FIGURE_CULTURES))
    ax.set_yticklabels(['all', ] + list(FIGURE_CULTURES))
    ax.set_ylim((-0.5, 5.5))
    ax.invert_yaxis()



def prettify_boxplot(ax, bplot):
    colors = ['green', 'gray']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_color(color)
    plt.setp(bplot['medians'], color='black')
    plt.setp(bplot['whiskers'], color='black')
    plt.xlabel('effective connection')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['yes', 'no'])
    adjust_position(ax, yshrink=0.02, xshrink=0.03, xshift=0.02)
    without_spines_and_ticks(ax)


def DataFrame_from_Dicts(structural_delay, structural_strength, synapse_delay):
    data = pd.merge(pd.merge(pd.DataFrame(((pre, post, value) for (pre, post), value in structural_delay.items()),
                                          columns=['pre', 'post', 'structural_delay']),
                             pd.DataFrame(((pre, post, value) for (pre, post), value in structural_strength.items()),
                                          columns=['pre', 'post', 'structural_strength']),
                             on=['pre', 'post']),
                    pd.DataFrame(((pre, post, value) for (pre, post), value in synapse_delay.items()),
                                 columns=['pre', 'post', 'synaptic_delay']),
                    on=['pre', 'post'], how='outer')
    data.fillna(0, inplace=True)   # zero synaptic delay for the connection not assigned a synapse
    data['delayed'] = data.synaptic_delay > 1  # ms
    data['simultaneous'] = data.synaptic_delay < 1  # ms
    return data


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))