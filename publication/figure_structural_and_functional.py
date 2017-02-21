import logging
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from hana.misc import unique_neurons
from hana.plotting import plot_axon, plot_dendrite, plot_neuron_points, plot_neuron_id, plot_neuron_pair, plot_network, mea_axes, highlight_connection
from hana.recording import load_positions, average_electrode_area
from hana.segmentation import load_compartments, load_neurites, neuron_position_from_trigger_electrode
from hana.structure import find_overlap, all_overlaps
from hana.function import timelag_standardscore, all_peaks
from publication.plotting import show_or_savefig, FIGURE_ARBORS_FILE, adjust_position, without_spines_and_ticks
from publication.figure_functional import plot_func_example_and_network, detect_function_networks, plot_std_score_and_peaks
from publication.figure_structural import plot_two_colorbars

logging.basicConfig(level=logging.DEBUG)


# Final version

def make_figure(figurename, figpath=None,
                presynaptic_neuron=5,
                postsynaptic_neuron=10,
                postsynaptic_neuron2 = 49,  # or 50
                thr_overlap_area = 3000.,  # um2/electrode
                thr_z_score = 10):

    if not os.path.isfile('temp/standardscores.p'):
        detect_function_networks()

    # Neuron positions
    trigger, _, axon_delay, dendrite_peak = load_compartments(FIGURE_ARBORS_FILE)
    pos = load_positions(mea='hidens')
    electrode_area = average_electrode_area(pos)
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)

    # Making figure
    fig = plt.figure(figurename, figsize=(14, 14))
    fig.suptitle(figurename + ' Estimate structural and functional connectivity', fontsize=14, fontweight='bold')


    # Examples for structual connected and unconnected neurons and structural network
    thr_overlap = np.ceil(thr_overlap_area / electrode_area)  # number of electrodes
    logging.info('Overlap of at least %d um2 corresponds to %d electrodes.' % (thr_overlap_area, thr_overlap))
    overlap, ratio, delay = find_overlap(axon_delay, dendrite_peak, presynaptic_neuron, postsynaptic_neuron,
                                         thr_overlap=thr_overlap)
    overlap2, ratio2, delay2 = find_overlap(axon_delay, dendrite_peak, presynaptic_neuron, postsynaptic_neuron2,
                                         thr_overlap=thr_overlap)

    ax1 = plt.subplot(421)
    plot_neuron_pair(ax1, pos, axon_delay, dendrite_peak, neuron_pos, postsynaptic_neuron, presynaptic_neuron, delay)
    mea_axes(ax1, style='off')
    ax1.set_title ('neuron pair %d $\longrightarrow$ %d' % (presynaptic_neuron, postsynaptic_neuron))
    plot_two_colorbars(ax1)
    adjust_position(ax1, yshrink=0.01)

    ax2 = plt.subplot(423)
    plot_neuron_pair(ax2, pos, axon_delay, dendrite_peak, neuron_pos, postsynaptic_neuron2, presynaptic_neuron, delay2)
    mea_axes(ax2, style='off')
    ax2.set_title('neuron pair %d $\dashrightarrow$ %d' % (presynaptic_neuron, postsynaptic_neuron2))
    plot_two_colorbars(ax2)
    adjust_position(ax2, yshrink=0.01)

    # Whole network
    ax3 = plt.subplot(222)
    all_overlap, all_ratio, all_delay = all_overlaps(axon_delay, dendrite_peak, thr_overlap=thr_overlap)
    plot_neuron_points(ax3, unique_neurons(all_delay), neuron_pos)
    plot_neuron_id(ax3, trigger, neuron_pos)
    plot_network (ax3, all_delay, neuron_pos)
    highlight_connection(ax3, (presynaptic_neuron, postsynaptic_neuron2), neuron_pos, connected=False)
    highlight_connection(ax3, (presynaptic_neuron, postsynaptic_neuron), neuron_pos)
    ax3.text(200,250,r'$\mathsf{\rho=%3d\mu\ m^2}$' % thr_overlap_area, fontsize=14)

    mea_axes(ax3)
    ax3.set_title ('structural connectivity graph')


    # Examples for functional connected and unconnected neurons and functional network

    timeseries, timeseries_surrogates = pickle.load(open( 'temp/timeseries_and_surrogates.p','rb'))
    timelags, std_score_dict, timeseries_hist_dict = pickle.load(open( 'temp/standardscores.p','rb'))
    peak_score, peak_timelag, _, _ = all_peaks(timelags, std_score_dict, thr=thr_z_score, direction='reverse')

    # Plotting reverse
    ax4 = plt.subplot(224)
    plot_network(ax4, peak_score, neuron_pos)
    neuron_dict = unique_neurons(peak_score)
    plot_neuron_points(ax4, neuron_dict, neuron_pos)
    plot_neuron_id(ax4, neuron_dict, neuron_pos)
    highlight_connection(ax4, (presynaptic_neuron, postsynaptic_neuron2), neuron_pos, connected=False)
    highlight_connection(ax4, (presynaptic_neuron, postsynaptic_neuron), neuron_pos)
    ax4.text(200,250,r'$\mathsf{\zeta=%d}$' % thr_z_score, fontsize=14)
    mea_axes(ax4)

    ax5 = plt.subplot(425)
    plot_z_score (ax5, presynaptic_neuron, postsynaptic_neuron, thr_z_score, peak_timelag, timeseries, timeseries_surrogates)
    ax5.set_title ('neuron pair %d $\longrightarrow$ %d' % (presynaptic_neuron, postsynaptic_neuron))
    without_spines_and_ticks(ax5)

    ax6 = plt.subplot(427)
    plot_z_score (ax6, presynaptic_neuron, postsynaptic_neuron2, thr_z_score, peak_timelag, timeseries, timeseries_surrogates)
    ax6.set_title ('neuron pair %d $\longrightarrow$ %d' % (presynaptic_neuron, postsynaptic_neuron2))
    without_spines_and_ticks(ax6)

    # plot_func_example_and_network(ax4, ax5, ax6, presynaptic_neuron, postsynaptic_neuron, 'reverse', thr, neuron_pos, std_score_dict,
    #                               timelags, timeseries, timeseries_surrogates)
    adjust_position(ax5, yshrink=0.01, xshrink=0.04)
    adjust_position(ax6, yshrink=0.01, xshrink=0.04)
    ax4.set_title('functional connectivity graph')
    ax5.set_ylim((-thr_z_score, thr_z_score*3))
    ax6.set_ylim((-thr_z_score, thr_z_score*3))

    fig.text(0.11, 0.9, 'A', size=30, weight='bold')
    fig.text(0.11, 0.69, 'B', size=30, weight='bold')
    fig.text(0.11, 0.48, 'D', size=30, weight='bold')
    fig.text(0.11, 0.27, 'E', size=30, weight='bold')
    fig.text(0.52, 0.9, 'C', size=30, weight='bold')
    fig.text(0.52, 0.46, 'F', size=30, weight='bold')

    show_or_savefig(figpath, figurename)

def plot_z_score (ax3, pre, post, thr, peak_timelag, timeseries, timeseries_surrogates):
    timelags, std_score, timeseries_hist, surrogates_mean, surrogates_std \
        = timelag_standardscore(timeseries[post], timeseries[pre],
                                timeseries_surrogates[pre])  # calculate for network
    if (pre, post) in peak_timelag:
        peak = -peak_timelag[(pre, post)]
    else:
        peak = None
    plot_std_score_and_peaks(ax3, timelags, std_score, thr=thr, peak=peak, loc=0)


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))

