import logging
import os
import numpy as np
from matplotlib import pyplot as plt

from hana.misc import unique_neurons
from hana.plotting import plot_axon, plot_dendrite, plot_neuron_points, plot_neuron_id, plot_neuron_pair, plot_network, mea_axes, highlight_connection
from hana.recording import load_positions, average_electrode_area
from hana.segmentation import neuron_position_from_trigger_electrode
from hana.structure import find_overlap, all_overlaps

from publication.data import Experiment, FIGURE_CULTURE, FIGURE_NEURON, FIGURE_CONNECTED_NEURON,\
    FIGURE_NOT_CONNECTED_NEURON, FIGURE_THRESHOLD_OVERLAP_AREA
from publication.plotting import show_or_savefig, label_subplot, adjust_position

logging.basicConfig(level=logging.DEBUG)


# Other plots, mainly used for finding good examples

def test_plot_all_axonal_fields():
    """Display axonal field for each neuron, good examples are 3605, 9952, 3093"""
    axon_delay, dendrite_peak = Experiment(FIGURE_CULTURE).neurites()
    pos = load_positions(mea='hidens')
    for neuron in axon_delay :
        ax=plt.subplot(111)
        plot_axon(ax, pos, axon_delay[neuron])
        mea_axes(ax)
        ax.set_title ('axon for neuron %d' % neuron)
        plt.show()


def test_plot_all_dendritic_fields_vs_one_axonal_field (presynaptic_neuron):
    """Display one axon and dendritic field for each neuron, good examples are 5-->10"""
    axon_delay, dendrite_peak = Experiment(FIGURE_CULTURE).neurites()
    pos = load_positions(mea='hidens')
    for postsynaptic_neuron in dendrite_peak :
        ax=plt.subplot(111)
        plot_axon(ax, pos, axon_delay[presynaptic_neuron])
        plot_dendrite(ax, pos, dendrite_peak[postsynaptic_neuron])
        mea_axes(ax)
        ax.set_title ('axon for neuron %d, dendrite for neuron %d' % (presynaptic_neuron, postsynaptic_neuron))
        plt.show()


def test_plot_only_overlap():
    """Plot overlap between axonal and dendritic fields of a presumably pre- and post-synaptic neuron pair"""
    trigger, _, axon_delay, dendrite_peak = Experiment(FIGURE_CULTURE).compartments()
    pos = load_positions(mea='hidens')
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)
    presynaptic_neuron = 5  # electrode 3605
    postsynaptic_neuron = 10  # electrode 4972
    delay = 2.3  # NOTE Arbitrary used for checking plot function
    plt.figure('Figure 6', figsize=(16, 8))
    ax=plt.subplot(111)
    plot_neuron_pair(ax, pos, axon_delay, dendrite_peak, neuron_pos, postsynaptic_neuron, presynaptic_neuron, delay)
    mea_axes(ax)
    ax.set_title ('neuron pair %d $\longrightarrow$ %d' % (presynaptic_neuron, postsynaptic_neuron))
    plt.show()


# Final version

def make_figure(figurename, figpath=None):
    trigger, _, axon_delay, dendrite_peak = Experiment(FIGURE_CULTURE).compartments()
    pos = load_positions(mea='hidens')
    electrode_area = average_electrode_area(pos)
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)

    thr_overlap = np.ceil(FIGURE_THRESHOLD_OVERLAP_AREA / electrode_area)  # number of electrodes
    logging.info('Overlap of at least %d um2 corresponds to %d electrodes.' % (FIGURE_THRESHOLD_OVERLAP_AREA, thr_overlap))

    overlap, ratio, delay = find_overlap(axon_delay, dendrite_peak, FIGURE_NEURON, FIGURE_CONNECTED_NEURON,
                                         thr_overlap=thr_overlap)
    overlap2, ratio2, delay2 = find_overlap(axon_delay, dendrite_peak, FIGURE_NEURON, FIGURE_NOT_CONNECTED_NEURON,
                                         thr_overlap=thr_overlap)

    # Making figure
    fig = plt.figure(figurename, figsize=(16, 9))
    fig.suptitle(figurename + ' Structural connectivity', fontsize=14, fontweight='bold')

    ax1 = plt.subplot(221)
    plot_neuron_pair(ax1, pos, axon_delay, dendrite_peak, neuron_pos, FIGURE_CONNECTED_NEURON, FIGURE_NEURON, delay)
    mea_axes(ax1)
    ax1.set_title ('neuron pair %d $\longrightarrow$ %d' % (FIGURE_NEURON, FIGURE_CONNECTED_NEURON))
    ax1.text(200,200,r'$\rho=$%3d$\mu m^2$' % FIGURE_THRESHOLD_OVERLAP_AREA)
    plot_two_colorbars(ax1)
    adjust_position(ax1, yshrink=0.01)
    label_subplot(ax1, 'A', xoffset=-0.005, yoffset=-0.01)

    ax2 = plt.subplot(223)
    plot_neuron_pair(ax2, pos, axon_delay, dendrite_peak, neuron_pos, FIGURE_NOT_CONNECTED_NEURON, FIGURE_NEURON, delay2)
    mea_axes(ax2)
    ax2.set_title('neuron pair %d $\dashrightarrow$ %d' % (FIGURE_NEURON, FIGURE_NOT_CONNECTED_NEURON))
    ax2.text(200,200,r'$\rho=$%3d$\mu m^2$' % FIGURE_THRESHOLD_OVERLAP_AREA)
    plot_two_colorbars(ax2)
    adjust_position(ax2, yshrink=0.01)
    label_subplot(ax2, 'B', xoffset=-0.005, yoffset=-0.01)

    # Whole network
    ax3 = plt.subplot(122)
    all_overlap, all_ratio, all_delay = all_overlaps(axon_delay, dendrite_peak, thr_overlap=thr_overlap)
    plot_neuron_points(ax3, unique_neurons(all_delay), neuron_pos)
    plot_neuron_id(ax3, trigger, neuron_pos)
    plot_network (ax3, all_delay, neuron_pos)
    highlight_connection(ax3, (FIGURE_NEURON, FIGURE_CONNECTED_NEURON), neuron_pos)
    highlight_connection(ax3, (FIGURE_NEURON, FIGURE_NOT_CONNECTED_NEURON), neuron_pos, connected=False)
    ax3.text(200,150,r'$\rho=$%3d$\mu m^2$' % FIGURE_THRESHOLD_OVERLAP_AREA)
    mea_axes(ax3)
    ax3.set_title ('structural connectivity graph')
    label_subplot(ax3, 'C', xoffset=-0.05, yoffset=-0.05)

    show_or_savefig(figpath, figurename)


def plot_two_colorbars(ax1):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="10%", pad=0.2)
    cax2 = divider.append_axes("left", size="10%", pad=1.0)
    import matplotlib as mpl
    cb1 = mpl.colorbar.ColorbarBase(cax1, cmap=plt.cm.summer, norm=plt.Normalize(vmin=0, vmax=2),
                                    orientation='vertical')
    cb1.set_label(r'$\mathsf{\tau_{axon}\ [ms]}$', fontsize=14)
    cb2 = mpl.colorbar.ColorbarBase(cax2, cmap=plt.cm.gray_r, norm=plt.Normalize(vmin=0, vmax=50),
                                    orientation='vertical')
    cb2.set_label(r'$\mathsf{V_p\ [\mu V]}$', fontsize=14)
    cb2.ax.yaxis.set_ticks_position('left')
    cb2.ax.yaxis.set_label_position('left')


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))
    test_plot_all_axonal_fields()
    test_plot_all_dendritic_fields_vs_one_axonal_field(5)
    test_plot_only_overlap()
