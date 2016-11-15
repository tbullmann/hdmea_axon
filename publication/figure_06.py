from hana.matlab import load_neurites, load_positions
from hana.structure import find_overlap, all_overlaps
from hana.plotting import plot_axon, plot_dendrite, plot_neuron_pair, plot_network, set_axis_hidens
from publication.plotting import FIGURE_ARBORS_FILE, FIGURE_ELECTRODES_FILE

from matplotlib import pyplot as plt


import logging
logging.basicConfig(level=logging.DEBUG)


# Other plots, mainly used for finding good examples

def test_plot_all_axonal_fields():
    """Display axonal field for each neuron, good examples are 3605, 9952, 3093"""
    axon_delay, dendrite_peak = load_neurites (FIGURE_ARBORS_FILE)
    pos = load_positions(FIGURE_ELECTRODES_FILE)
    for neuron in axon_delay :
        ax=plt.subplot(111)
        plot_axon(ax, pos, axon_delay[neuron])
        set_axis_hidens(ax,pos)
        ax.set_title ('axon for neuron %d' % neuron)
        plt.show()


def test_plot_all_dendritic_fields_vs_one_axonal_field (presynaptic_neuron):
    """Display dendritic field for each neuron,
    good examples are 3605->7374,5242,4972,not 6271,6294,8060; 9952->not3605,7374,4972,5242;3093->3605, not7374,5242,4972"""
    axon_delay, dendrite_peak = load_neurites (FIGURE_ARBORS_FILE)
    pos = load_positions(FIGURE_ELECTRODES_FILE)
    for postsynaptic_neuron in dendrite_peak :
        ax=plt.subplot(111)
        plot_axon(ax, pos, axon_delay[presynaptic_neuron])
        plot_dendrite(ax, pos, dendrite_peak[postsynaptic_neuron])
        set_axis_hidens(ax,pos)
        ax.set_title ('axon for neuron %d, dendrite for neuron %d' % (presynaptic_neuron, postsynaptic_neuron))
        plt.show()


# Early version

def Figure06_only_overlap():
    """Plot overlap between axonal and dendritic fields of a presumably pre- and post-synaptic neuron pair"""
    axon_delay, dendrite_peak = load_neurites (FIGURE_ARBORS_FILE)
    pos = load_positions(FIGURE_ELECTRODES_FILE)
    presynaptic_neuron = 3605
    postsynaptic_neuron = 4972
    delay = 2.3  # NOTE Arbitrary used for checking plot function
    plt.figure('Figure 6', figsize=(16, 8))
    ax=plt.subplot(111)
    plot_neuron_pair(ax, pos, axon_delay, dendrite_peak, delay, postsynaptic_neuron, presynaptic_neuron)
    set_axis_hidens(ax,pos)
    ax.set_title ('neuron pair %d $\longrightarrow$ %d' % (presynaptic_neuron, postsynaptic_neuron))
    plt.show()


# Final version

def Figure06():
    axon_delay, dendrite_peak = load_neurites (FIGURE_ARBORS_FILE)
    pos = load_positions(FIGURE_ELECTRODES_FILE)

    presynaptic_neuron = 3605
    postsynaptic_neuron = 4972
    thr_peak = 5
    thr_overlap = 0.05

    ratio, delay = find_overlap(axon_delay, dendrite_peak, presynaptic_neuron, postsynaptic_neuron, thr_peak, thr_overlap)

    plt.figure('Figure 6', figsize=(16, 8))
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


# test_plot_all_axonal_fields()
# test_plot_all_dendritic_fields_vs_one_axonal_field(3605)
# Figure06_only_overlap
Figure06()
