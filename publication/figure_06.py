import logging

import numpy as np
from matplotlib import pyplot as plt

from hana.matlab import load_neurites, load_positions
from hana.plotting import plot_axon, plot_dendrite, plot_neuron_pair
from hana.structure import find_overlap, all_overlaps
from hana.plotting import plot_network
from hana.plotting import set_axis_hidens, plot_neuron_points, plot_neuron_id, highlight_connection

logging.basicConfig(level=logging.DEBUG)


def test():
    axon_delay, dendrite_peak = load_neurites ('data/hidens2018at35C_arbors.mat')
    pos = load_positions('data/hidens_electrodes.mat')

    print len(axon_delay)
    print axon_delay.keys()

    # neuron indicies are 6272, 9861, 7050, 5396, 3094, 6295, 7965, 6533, 3241, 3606, 2874, 5697, 10052, 8901, 10569,
    # 7375, 10964, 5978, 7903, 9953, 7014, 6249, 4973, 5243, 8061]


    pre = 3241
    post =4973

    ax = plt.subplot(111)

    z = axon_delay[pre]
    print np.nanmax(z)

    plot_axon(ax, pos, z)

    # for neuron in dendrite_peak:
    #     z = dendrite_peak[neuron]
    #     plot_dendrite(ax,pos,z)

    z = dendrite_peak[post]
    plot_dendrite(ax, pos, z)
    plot_neuron_points(ax, axon_delay, pos)
    plot_neuron_id(ax, axon_delay, pos)
    highlight_connection(ax,(pre,post),pos)

    set_axis_hidens(ax,pos)


    plt.show()

def test_plot_all_axonal_fields():
    """Display axonal field for each neuron, good examples are 3606, 9953, 3094"""
    axon_delay, dendrite_peak = load_neurites ('data/hidens2018at35C_arbors.mat')
    pos = load_positions('data/hidens_electrodes.mat')
    for neuron in axon_delay :
        ax=plt.subplot(111)
        plot_axon(ax, pos, axon_delay[neuron])
        set_axis_hidens(ax,pos)
        ax.set_title ('axon for neuron %d' % neuron)
        plt.show()

def test_plot_all_dendritic_fields(presynaptic_neuron):
    """Display dendritic field for each neuron,
    good examples are 3606->7375,5243,4973,not 6272,6295,8061; 9953->not3606,7375,4973,5243;3094->3606, not7375,5243,4973"""
    axon_delay, dendrite_peak = load_neurites ('data/hidens2018at35C_arbors.mat')
    pos = load_positions('data/hidens_electrodes.mat')
    for postsynaptic_neuron in dendrite_peak :
        ax=plt.subplot(111)
        plot_axon(ax, pos, axon_delay[presynaptic_neuron])
        plot_dendrite(ax, pos, dendrite_peak[postsynaptic_neuron])
        set_axis_hidens(ax,pos)
        ax.set_title ('axon for neuron %d, dendrite for neuron %d' % (presynaptic_neuron, postsynaptic_neuron))
        plt.show()

def test_plot_overlap():
    """Plot overlap between axonal and dendritic fields of a presumably pre- and post-synaptic neuron pair"""
    axon_delay, dendrite_peak = load_neurites ('data/hidens2018at35C_arbors.mat')
    pos = load_positions('data/hidens_electrodes.mat')
    presynaptic_neuron = 3606
    postsynaptic_neuron = 4973  # 5243
    delay = 2.3

    ax=plt.subplot(111)

    plot_neuron_pair(ax, pos, axon_delay, dendrite_peak, delay, postsynaptic_neuron, presynaptic_neuron)

    set_axis_hidens(ax,pos)
    ax.set_title ('neuron pair %d $\longrightarrow$ %d' % (presynaptic_neuron, postsynaptic_neuron))
    plt.show()


def test_estimate_overlap():
    axon_delay, dendrite_peak = load_neurites ('data/hidens2018at35C_arbors.mat')
    pos = load_positions('data/hidens_electrodes.mat')

    presynaptic_neuron = 3606
    postsynaptic_neuron = 4973  # 5243
    thr_peak = 5
    thr_overlap = 0.05

    ratio, delay = find_overlap(axon_delay, dendrite_peak, presynaptic_neuron, postsynaptic_neuron, thr_peak, thr_overlap)

    ax1 = plt.subplot(121)
    plot_neuron_pair(ax1, pos, axon_delay, dendrite_peak, delay, postsynaptic_neuron, presynaptic_neuron)
    set_axis_hidens(ax1,pos)
    ax1.set_title ('neuron pair %d $\longrightarrow$ %d' % (presynaptic_neuron, postsynaptic_neuron))

    ax2 = plt.subplot(122)
    all_ratio, all_delay = all_overlaps(axon_delay, dendrite_peak)
    plot_network (ax2, all_delay, pos)
    set_axis_hidens(ax2,pos)
    ax2.set_title ('structural connectivity graph')
    plt.show()


# test()
test_plot_all_axonal_fields()
test_plot_all_dendritic_fields(3606)
test_plot_overlap() # plots examplary overlap between axon and dendrite
test_estimate_overlap()  # figure 6
# figure10_prepare_data()
