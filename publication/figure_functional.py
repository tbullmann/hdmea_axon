from hana.function import timeseries_to_surrogates, all_timelag_standardscore, all_peaks, timelag_standardscore
from hana.segmentation import load_compartments, neuron_position_from_trigger_electrode
from hana.recording import load_timeseries, load_positions
from hana.plotting import plot_network, plot_neuron_points, plot_neuron_id, mea_axes, \
    plot_timeseries_hist_and_surrogates, plot_std_score_and_peaks, highlight_connection
from hana.misc import unique_neurons
from publication.plotting import show_or_savefig, FIGURE_EVENTS_FILE, FIGURE_ARBORS_FILE, label_subplot, adjust_position

import pickle
import os
from matplotlib import pyplot as plt


def detect_function_networks():
    timeseries = load_timeseries(FIGURE_EVENTS_FILE)
    timeseries_surrogates = timeseries_to_surrogates(timeseries)
    timelags, std_score_dict, timeseries_hist_dict = all_timelag_standardscore(timeseries, timeseries_surrogates)

    pickle.dump((timeseries, timeseries_surrogates),open('temp/timeseries_and_surrogates.p','wb'))
    pickle.dump((timelags, std_score_dict, timeseries_hist_dict),open('temp/standardscores.p','wb'))


def plot_func_network_forward_vs_reverse(thr, pos, timelags, std_score_dict):
    """Display functional networks calculated either with pre-->post if post fired after pre (forward condition)
    or pre-->post if pre fired before post (reverse condition) """
    score_forward, _, _, _ = all_peaks(timelags, std_score_dict, thr=thr, direction='forward')
    score_reverse, _, _, _ = all_peaks(timelags, std_score_dict, thr=thr, direction='reverse')
    print "forward k = ", len(score_forward)
    print "reverse k = ", len(score_reverse)
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    plot_network(ax1, score_forward, pos)
    neuron_dict = unique_neurons(score_forward)
    plot_neuron_points(ax1, neuron_dict, pos)
    plot_neuron_id(ax1, neuron_dict, pos)
    mea_axes(ax1)
    ax1.set_title(r'pre$\longrightarrow$post if post fired after pre')
    plot_network(ax2, score_reverse, pos)
    neuron_dict = unique_neurons(score_reverse)
    plot_neuron_points(ax2, neuron_dict, pos)
    plot_neuron_id(ax2, neuron_dict, pos)
    mea_axes(ax2)
    ax2.set_title('pre$\longrightarrow$post if pre fired before post')
    mea_axes(ax2)


def plot_func_example_and_network(ax1, ax2, ax3, pre, post, direction, thr, pos, std_score_dict, timelags, timeseries,
                                  timeseries_surrogates):
    """Displays functional connectivity of one highlighted connection withing the network"""
    # Calculate (again) details for a single neuron pair
    if direction == 'forward':
        timelags, std_score, timeseries_hist, surrogates_mean, surrogates_std \
            = timelag_standardscore(timeseries[pre], timeseries[post], timeseries_surrogates[post])
    if direction == 'reverse':
        timelags, std_score, timeseries_hist, surrogates_mean, surrogates_std \
            = timelag_standardscore(timeseries[post], timeseries[pre],
                                    timeseries_surrogates[pre])  # calculate for network
    peak_score, peak_timelag, _, _ = all_peaks(timelags, std_score_dict, thr=thr, direction=direction)
    # Plot histograms for single neuron pair
    loc = 2 if direction=='forward' else 1
    plot_timeseries_hist_and_surrogates(ax2, timelags, timeseries_hist, surrogates_mean, surrogates_std,loc=loc)
    if direction == 'forward': ax2.set_title('neuron pair %d $\longrightarrow$ %d' % (pre, post), loc='left')
    if direction == 'reverse': ax2.set_title('neuron pair %d $\longleftarrow$ %d' % (post, pre), loc='left')

    if (pre, post) in peak_timelag:
        peak = peak_timelag[(pre, post)]
        if direction == 'reverse': peak = -peak
    else:
        peak = None
    plot_std_score_and_peaks(ax3, timelags, std_score, thr=thr, peak=peak, loc=loc)
    # Plot network and highlight the connection between the single neuron pair
    plot_network(ax1, peak_score, pos)
    neuron_dict = unique_neurons(peak_score)
    plot_neuron_points(ax1, neuron_dict, pos)
    plot_neuron_id(ax1, neuron_dict, pos)
    if direction == 'forward': ax1.set_title(r'pre-synaptic spike followed by post-synaptic spike')
    if direction == 'reverse': ax1.set_title(r'post-synaptic spike preceded by pre-synaptic spike ')
    if peak is not None: highlight_connection(ax1, (pre, post), pos)
    ax1.text(200,150,r'$\zeta=$%d' % thr)
    mea_axes(ax1, barposition='inside')


def make_figure(figurename, figpath=None, thr =20):
    """FIGURE showing Displays functional connectivity according to forward and reverse definition for two
    neuron pairs within the network"""
    if not os.path.isfile('temp/standardscores.p'):
        detect_function_networks()

    timeseries, timeseries_surrogates = pickle.load(open( 'temp/timeseries_and_surrogates.p','rb'))
    timelags, std_score_dict, timeseries_hist_dict = pickle.load(open( 'temp/standardscores.p','rb'))

    trigger, _, axon_delay, dendrite_peak = load_compartments(FIGURE_ARBORS_FILE)
    pos = load_positions(mea='hidens')
    neuron_pos = neuron_position_from_trigger_electrode (pos, trigger)

    # for k,v in trigger.iteritems(): print (k,v)  # show neuron -> trigger electrode index
    pre, post  = 10, 4 # electrodes 4972, 3240
    # pre, post = 37, 31 #  pre, post = 8060,7374

    # Making figure
    fig = plt.figure(figurename, figsize=(16, 16))
    fig.suptitle(figurename + ' Functional connectivity', fontsize=14, fontweight='bold')

    # Plotting forward
    ax1 = plt.subplot(222)
    ax2 = plt.subplot(421)
    ax3 = plt.subplot(423)
    plot_func_example_and_network(ax1, ax2, ax3, pre, post, 'forward', thr, neuron_pos, std_score_dict,
                                  timelags, timeseries, timeseries_surrogates)
    adjust_position(ax2, yshrink=0.01)
    adjust_position(ax3, yshrink=0.01)
    label_subplot(ax1, 'C', xoffset=-0.03, yoffset=-0.01)
    label_subplot(ax2, 'A', xoffset=-0.05, yoffset=-0.01)
    label_subplot(ax3, 'B', xoffset=-0.05, yoffset=-0.01)

    # Plotting reverse
    ax4 = plt.subplot(224)
    ax5 = plt.subplot(425)
    ax6 = plt.subplot(427)
    plot_func_example_and_network(ax4, ax5, ax6, pre, post, 'reverse', thr, neuron_pos, std_score_dict,
                                  timelags, timeseries, timeseries_surrogates)
    adjust_position(ax5, yshrink=0.01)
    adjust_position(ax6, yshrink=0.01)
    label_subplot(ax4, 'F', xoffset=-0.03, yoffset=-0.01)
    label_subplot(ax5, 'D', xoffset=-0.05, yoffset=-0.01)
    label_subplot(ax6, 'E', xoffset=-0.05, yoffset=-0.01)

    show_or_savefig(figpath, figurename)


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))
