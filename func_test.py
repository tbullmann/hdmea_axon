import pickle

import matplotlib.pyplot as plt
import numpy as np

from func import timeseries_to_surrogates, all_timelag_standardscore, timelag_by_for_loop, timelag_by_sawtooth, \
    timelag_hist, timelag, randomize_intervals_by_swapping, randomize_intervals_by_gaussian, surrogate_timeseries, \
    timelag_standardscore, find_peaks, all_peaks, swap_intervals
from mio import load_events, events_to_timeseries, load_positions
from plot_network import set_axis_hidens, unique_neurons, plot_neuron_id, plot_neuron_points, plot_network, \
    highlight_connection, plot_pair_func, plot_std_score_and_peaks, plot_timeseries_hist_and_surrogates


def test_swap_intervals ():
    timeseries = np.array([1,2,4,5,7,8,10])
    indicies = np.array([0, 4])
    print timeseries
    print indicies
    print swap_intervals(timeseries,indicies)

def test_timelag(n):
    timeseries1 = np.array(range(n))
    timeseries2 = timeseries1
    print timeseries1, " -> ", timeseries2
    print "timelags by for loop = ", timelag_by_for_loop(timeseries1, timeseries2)
    print "timelags by sawtooth = ", timelag_by_sawtooth(timeseries1, timeseries2)
    timeseries2 = timeseries1+0.1
    print timeseries1, " -> ", timeseries2
    print "timelags by for loop = ", timelag_by_for_loop(timeseries1, timeseries2)
    print "timelags by sawtooth = ", timelag_by_sawtooth(timeseries1, timeseries2)
    timeseries2 = timeseries1-0.2
    print timeseries1, " -> ", timeseries2
    print "timelags by for loop = ", timelag_by_for_loop(timeseries1, timeseries2)
    print "timelags by sawtooth = ", timelag_by_sawtooth(timeseries1, timeseries2)

def test_timelag_hist (n):
    timeseries1 = np.sort(np.random.rand(1, n))[0]
    timeseries2 = np.sort(np.random.rand(1, n))[0]
    print timelag_hist(timelag(timeseries1, timeseries2))[0]

def test_randomize_intervals (n, factor=2):
    timeseries = np.array(np.cumsum(range(n)))
    print "original timeseries    = ", timeseries
    print "gaps                   = ", np.diff(timeseries)
    print "randomized by swapping = ", np.diff(randomize_intervals_by_swapping(timeseries,factor))
    print "randomized by gaussian = ", np.diff(randomize_intervals_by_gaussian(timeseries,factor))

def test_surrogates (n):
    timeseries1 =  np.sort(np.random.rand(n+1))
    timeseries2 = np.sort(0.0002*np.random.rand(n+1) + np.copy(timeseries1)+0.002)
    surrogates = surrogate_timeseries(timeseries2, n=20)
    timelags, std_score, timeseries_hist, surrogates_mean, surrogates_std = timelag_standardscore(timeseries1,
                                                                                         timeseries2, surrogates)
    print "Score: Done."

    score_max, timelags_max, timelag_min, timelag_max = find_peaks (timelags, std_score, thr=10)
    print "Peaks: Done"

    if len(score_max)>0:
        print "peak score   =", score_max[0]
        print "peak timelag =", timelag_max[0]

    plot_pair_func(timelags, timeseries_hist, surrogates_mean, surrogates_std, std_score, 'Testing surrogate timeseries')
    plt.show()


def load_test_data():
    events = load_events ('data/hidens2018at35C_events.mat')
    return events

def test_on_data1():
    events = load_test_data()
    timeseries = events_to_timeseries(events)
    timeseries_surrogates = timeseries_to_surrogates(timeseries)
    timelags, std_score_dict, timeseries_hist_dict = all_timelag_standardscore(timeseries, timeseries_surrogates)

    import matplotlib.pyplot as plt
    for pair in std_score_dict:
        plt.plot(timelags*1000, std_score_dict[pair])
    plt.show()

    pickle.dump((timeseries, timeseries_surrogates),open( 'temp/timeseries_and_surrogates_hidens2018.p','wb'))
    pickle.dump((timelags, std_score_dict, timeseries_hist_dict),open( 'temp/standardscores_hidens2018.p','wb'))


def test_on_data2(thr =20):
    """FIGURE showing functional networks according to forward and reverse definition"""
    timelags, std_score_dict, timeseries_hist_dict = pickle.load(open( 'temp/standardscores_hidens2018.p','rb'))
    pos = load_positions('data/hidens_electrodes.mat')
    plot_func_network_forward_vs_reverse(thr, pos, timelags, std_score_dict)
    plt.show()


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
    set_axis_hidens(ax1, pos)
    ax1.set_title(r'pre$\longrightarrow$post if post fired after pre')
    plot_network(ax2, score_reverse, pos)
    neuron_dict = unique_neurons(score_reverse)
    plot_neuron_points(ax2, neuron_dict, pos)
    plot_neuron_id(ax2, neuron_dict, pos)
    set_axis_hidens(ax2, pos)
    ax2.set_title('pre$\longrightarrow$post if pre fired before post')
    set_axis_hidens(ax2, pos)


def test_on_data3(thr =20):
    """FIGURE showing Displays functional connectivity according to forward and reverse definition for two
    neuron pairs within the network"""
    timeseries, timeseries_surrogates = pickle.load(open( 'temp/timeseries_and_surrogates_hidens2018.p','rb'))
    timelags, std_score_dict, timeseries_hist_dict = pickle.load(open( 'temp/standardscores_hidens2018.p','rb'))
    pos = load_positions('data/hidens_electrodes.mat')
    for pre,post in ((4972,3240), (8060,7374)):
        for direction in ('forward', 'reverse'):
            plot_func_example_and_network(pre, post, direction, thr, pos, std_score_dict, timelags, timeseries,
                                          timeseries_surrogates)
            plt.show()


def plot_func_example_and_network(pre, post, direction, thr, pos, std_score_dict, timelags, timeseries,
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
    # Plotting
    plt.figure(figsize=(20, 10))
    # Plot histograms for single neuron pair
    ax2 = plt.subplot(221)
    plot_timeseries_hist_and_surrogates(ax2, timelags, timeseries_hist, surrogates_mean, surrogates_std)
    if direction == 'forward': ax2.set_title('%d $\longrightarrow$ %d' % (pre, post))
    if direction == 'reverse': ax2.set_title('%d $\longleftarrow$ %d' % (pre, post))
    ax3 = plt.subplot(223)
    if (pre, post) in peak_timelag:
        peak = peak_timelag[(pre, post)]
        if direction == 'reverse': peak = -peak
    else:
        peak = None
    plot_std_score_and_peaks(ax3, timelags, std_score, thr=thr, peak=peak)
    # Plot network and highlight the connection between the single neuron pair
    ax1 = plt.subplot(122)
    plot_network(ax1, peak_score, pos)
    neuron_dict = unique_neurons(peak_score)
    plot_neuron_points(ax1, neuron_dict, pos)
    plot_neuron_id(ax1, neuron_dict, pos)
    set_axis_hidens(ax1, pos)
    if direction == 'forward': ax1.set_title(r'pre$\longrightarrow$post if post fired after pre')
    if direction == 'reverse': ax1.set_title('pre$\longrightarrow$post if pre fired before post')
    if peak is not None: highlight_connection(ax1, (pre, post), pos)
    set_axis_hidens(ax1, pos)


# test_timelag (5)
# test_timelag_hist(10000)
# test_randomize_intervals(10)
# test_surrogates(1000)
# test_on_data1()
# test_on_data2()
# test_on_data3()


