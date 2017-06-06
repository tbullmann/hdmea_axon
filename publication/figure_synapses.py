from __future__ import division

import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hana.misc import unique_neurons
from hana.plotting import plot_network, plot_neuron_points, mea_axes, plot_neuron_id
from hana.recording import average_electrode_area
from hana.segmentation import load_positions, neuron_position_from_trigger_electrode
from publication.data import Experiment, FIGURE_CULTURE, FIGURE_CULTURES
from publication.plotting import show_or_savefig, plot_synapse_delays, adjust_position, \
    without_spines_and_ticks

logging.basicConfig(level=logging.DEBUG)


def make_figure(figurename, figpath=None):

    structural_strength, structural_delay, functional_strength, functional_delay, synapse_strength, synapse_delay \
        = Experiment(FIGURE_CULTURE).networks()

    timelags, std_score_dict, timeseries_hist_dict = Experiment(FIGURE_CULTURE).standardscores()
    minimal_synaptic_delay = 1.0

    # Map electrode number to area covered by that electrodes
    electrode_area = average_electrode_area(None, mea='hidens')

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 13))
    if not figpath:
        fig.suptitle(figurename + ' Estimate synaptic connectivity', fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.05)

    remaining_keys = [key for key in structural_delay.keys() if not key in synapse_delay.keys()]

    ax12 = plt.subplot(421)
    postsynaptic = timelags > 0
    for i, (pair, axonal_delay) in enumerate(sorted(structural_delay.items(), key=lambda x: x[1])):
        x = timelags[postsynaptic] - axonal_delay
        y = i + std_score_dict[pair][postsynaptic]
        xx = np.hstack((x[0], np.ravel(zip(x[:-1], x[1:])) , x[-1]))
        yy = np.hstack((i,    np.ravel(zip(y[:-1], y[:-1])), i))
        plt.fill_between(xx, 0, yy, color='white', zorder=-i, clip_on=False)
        plt.plot(xx, yy, color='black', zorder=-i, clip_on=False)
        plt.plot(-axonal_delay, i, 'b.')
        if pair in synapse_delay:
            plt.vlines(synapse_delay[pair] - axonal_delay +0.05, i, i+synapse_strength[pair], 'green', linewidth=5, zorder=-i)
    without_spines_and_ticks(ax12)
    ax12.set_ylim((-10,250))
    ax12.set_xlim((-2.2,5))
    ax12.spines['bottom'].set_bounds(-2, 5)
    ax12.spines['left'].set_bounds(0, 100)
    ax12.set_yticks([0,50,100])
    ax12.set_yticklabels([0,50,100])
    ax12.set_ylabel(r'$\mathsf{z}$', fontsize=14)
    ax12.yaxis.set_label_coords(-0.1, 0.25)
    ax12.set_xlabel(r'$\mathsf{\tau_{synapse}=\tau_{spike}-\tau_{axon}\ [ms]}$', fontsize=14)
    plt.title('a', loc='left', fontsize=18)

    ax111 = plt.subplot(423)
    for i, (pair, axonal_delay) in enumerate(sorted(structural_delay.items(), key=lambda x: x[1])):
        plt.plot(-axonal_delay, i, 'b.')
        if pair in synapse_delay:
            plt.plot(synapse_delay[pair]-axonal_delay,i, 'g.')
    ax111.set_ylabel(r'$\mathsf{structural\ connection\ rank\ by\ \tau_{axon}}$', fontsize=14)
    ax111.fill([0, minimal_synaptic_delay, minimal_synaptic_delay, 0], [0, 0, 250, 250], fill=False, hatch='\\',
               linewidth=0)
    ax111.set_ylim((0, 220))
    ax111.set_xticks([-2, -1, 0, 1, 2, 3, 4, 5])
    ax111.set_xticklabels([2, 1, 0, 1, 2, 3, 4, 5])
    without_spines_and_ticks(ax111)
    ax111.set_xlabel(r'$\mathsf{\tau_{axon}\ [ms]\ \leftarrow\ \ \ \ \ \rightarrow\tau_{synapse}\ [ms]}$', fontsize=14)
    ax111.xaxis.set_label_coords(0.31, -0.1)
    ax111.spines['bottom'].set_bounds(-2, 5)
    ax111.set_xlim((-2.2,5))
    adjust_position(ax111, yshrink=0.01, yshift=-0.01)
    plt.title('b', loc='left', fontsize=18)

    ax2 = plt.subplot(222)
    trigger, _, _, _ = Experiment(FIGURE_CULTURE).compartments()
    pos = load_positions(mea='hidens')
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)
    plot_network(ax2, synapse_delay.keys(), neuron_pos, color='gray')
    plot_network(ax2, remaining_keys, neuron_pos, color='green')
    plot_neuron_points(ax2, unique_neurons(structural_delay), neuron_pos)
    plot_neuron_id(ax2, trigger, neuron_pos)
    # Legend by proxy
    ax2.hlines(0, 0, 0, linestyle='-', color='gray', label='none')
    ax2.hlines(0, 0, 0, linestyle='-', color='green', label='synapse')
    plt.legend(frameon=False)
    mea_axes(ax2)
    adjust_position(ax2, yshift=-0.01)
    plt.title('c     synaptic connectivity graph', loc='left', fontsize=18)

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

    ax3 = plt.subplot(256)
    bplot = ax3.boxplot(np.array(connections[['delayed','simultaneous']]),
                        boxprops={'color': "black", "linestyle": "-"}, patch_artist=True,widths=0.7)
    prettify_boxplot(ax3, bplot)
    ax3.set_ylim((0,100))
    ax3.set_ylabel('percent of structural connections [%]')
    adjust_position(ax3, xshift=-0.02)
    plt.title('d', loc='left', fontsize=18)

    values = pd.pivot_table(data, index='culture', columns='delayed',
                        values=['structural_strength', 'structural_delay'], aggfunc=np.median)

    ax4 = plt.subplot(257)
    bplot = ax4.boxplot(np.array(values['structural_strength']) * electrode_area,   # overlap in um2
                        boxprops={'color': "black", "linestyle": "-"}, patch_artist=True, widths=0.7)
    prettify_boxplot(ax4, bplot)
    ax4.set_ylabel(r'$\mathsf{|A \cap D|\ [\mu m^2]}$', fontsize=14)
    plt.title('e', loc='left', fontsize=18)

    ax5 = plt.subplot(258)
    bplot = ax5.boxplot(np.array(values['structural_delay']),
                        boxprops={'color': "black", "linestyle": "-"}, patch_artist=True, widths=0.7)
    prettify_boxplot(ax5, bplot)
    ax5.set_ylim((0,2))
    ax5.set_ylabel(r'$\mathsf{\tau_{axon}\ [ms]}$', fontsize=14)
    plt.title('f', loc='left', fontsize=18)

    ax6 = plt.subplot(259)
    # show distirbution of median synapse delays as boxplot
    bplot = plt.boxplot([np.median(x) for x in synapse_delays], 0, '+', 0, positions=np.array([0.]),
                        boxprops={'color': "black", "linestyle": "-"}, patch_artist=True, widths=0.7)
    bplot['boxes'][0].set_color('green')
    plt.setp(bplot['medians'], color='black')
    plt.setp(bplot['whiskers'], color='black')
    # add violin plot for every culture
    vplot = plt.violinplot(synapse_delays, FIGURE_CULTURES, points=80, vert=False, widths=0.7,
                           showmeans=False, showextrema=True, showmedians=True)
    ax6.fill([0, minimal_synaptic_delay, minimal_synaptic_delay, 0], [-1, -1, 9, 9],
             fill=False, hatch='\\', linewidth=0)

    # Make all the violin statistics marks black:
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = vplot[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
    # Make the violin body blue with a black border:
    for pc in vplot['bodies']:
        pc.set_facecolor('green')
        pc.set_alpha(0.5)
        pc.set_edgecolor('none')
    ax6.set_xlabel(r'$\mathsf{\tau_{synapse}\ [ms]}$', fontsize=14)
    ax6.set_xticks([0, 1, 2, 3, 4, 5])
    ax6.set_xticklabels([0, 1, 2, 3, 4, 5])
    ax6.set_ylabel('culture')
    ax6.set_yticks([0,]+list(FIGURE_CULTURES))
    ax6.set_yticklabels(['all',]+list(FIGURE_CULTURES))
    ax6.set_ylim((-0.5,6.5))
    ax6.invert_yaxis()
    adjust_position(ax6, xshift=0.08, xshrink=-0.05, yshrink=0.04)
    without_spines_and_ticks(ax6)
    plt.title('g', loc='left', fontsize=18)

    show_or_savefig(figpath, figurename)


def prettify_boxplot(ax, bplot):
    colors = ['green', 'gray']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_color(color)
    plt.setp(bplot['medians'], color='black')
    plt.setp(bplot['whiskers'], color='black')
    plt.xlabel('putative synapse')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['yes', 'no'])
    adjust_position(ax,xshrink=0.02, yshrink=0.04)
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