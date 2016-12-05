import logging

from matplotlib import pyplot as plt

from hana.misc import unique_neurons
from hana.plotting import plot_axon, plot_dendrite, plot_neuron_points, plot_neuron_id, plot_neuron_pair, plot_network, set_axis_hidens
from hana.recording import load_positions, HIDENS_ELECTRODES_FILE
from hana.structure import find_overlap, all_overlaps, load_neurites, load_compartments, \
    neuron_position_from_trigger_electrode
from publication.plotting import FIGURE_ARBORS_FILE

logging.basicConfig(level=logging.DEBUG)


# Other plots, mainly used for finding good examples

def test_plot_all_axonal_fields():
    """Display axonal field for each neuron, good examples are 3605, 9952, 3093"""
    axon_delay, dendrite_peak = load_neurites(FIGURE_ARBORS_FILE)
    pos = load_positions(HIDENS_ELECTRODES_FILE)
    for neuron in axon_delay :
        ax=plt.subplot(111)
        plot_axon(ax, pos, axon_delay[neuron])
        set_axis_hidens(ax,pos)
        ax.set_title ('axon for neuron %d' % neuron)
        plt.show()


def test_plot_all_dendritic_fields_vs_one_axonal_field (presynaptic_neuron):
    """Display one axon and dendritic field for each neuron, good examples are 5-->10"""
    axon_delay, dendrite_peak = load_neurites (FIGURE_ARBORS_FILE)
    pos = load_positions(HIDENS_ELECTRODES_FILE)
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
    trigger, _, axon_delay, dendrite_peak = load_compartments (FIGURE_ARBORS_FILE)
    pos = load_positions(HIDENS_ELECTRODES_FILE)
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)
    presynaptic_neuron = 5  # electrode 3605
    postsynaptic_neuron = 10  # electrode 4972
    delay = 2.3  # NOTE Arbitrary used for checking plot function
    plt.figure('Figure 6', figsize=(16, 8))
    ax=plt.subplot(111)
    plot_neuron_pair(ax, pos, axon_delay, dendrite_peak, neuron_pos, postsynaptic_neuron, presynaptic_neuron, delay)
    set_axis_hidens(ax,pos)
    ax.set_title ('neuron pair %d $\longrightarrow$ %d' % (presynaptic_neuron, postsynaptic_neuron))
    plt.show()


# Final version

def Figure06():
    trigger, _, axon_delay, dendrite_peak = load_compartments(FIGURE_ARBORS_FILE)
    pos = load_positions(HIDENS_ELECTRODES_FILE)
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)

    presynaptic_neuron = 5 # electrode 3605
    postsynaptic_neuron = 10 #electrode 4972
    thr_peak = 5
    thr_overlap = 0.05

    ratio, delay = find_overlap(axon_delay, dendrite_peak, presynaptic_neuron, postsynaptic_neuron, thr_peak, thr_overlap)

    plt.figure('Figure 6', figsize=(12, 6))
    ax1 = plt.subplot(121)
    plot_neuron_pair(ax1, pos, axon_delay, dendrite_peak, neuron_pos, postsynaptic_neuron, presynaptic_neuron, delay)
    set_axis_hidens(ax1,pos)
    ax1.set_title ('neuron pair %d $\longrightarrow$ %d' % (presynaptic_neuron, postsynaptic_neuron))

    ax2 = plt.subplot(122)
    all_ratio, all_delay = all_overlaps(axon_delay, dendrite_peak)
    plot_neuron_points(ax2, unique_neurons(all_delay), neuron_pos)
    plot_neuron_id(ax2, trigger, neuron_pos)
    plot_network (ax2, all_delay, neuron_pos)
    set_axis_hidens(ax2, pos)
    ax2.set_title ('structural connectivity graph')
    plt.show()


# test_plot_all_axonal_fields()
# test_plot_all_dendritic_fields_vs_one_axonal_field(5)
# Figure06_only_overlap()
Figure06()
