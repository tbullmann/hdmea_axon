from __future__ import division
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import os

from publication.data import Experiment, FIGURE_CULTURES
from publication.simulation import Simulation
from publication.plotting import show_or_savefig, without_spines_and_ticks, adjust_position
from publication.figure_graphs import flatten
from publication.comparison import print_test_result
from publication import burst_detection as bd

import logging
logging.basicConfig(level=logging.DEBUG)



def make_figure(figurename, figpath=None):

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 10))
    if not figpath:
        fig.suptitle(figurename + '    Dynamics of original and simulated networks', fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.1)

    exp_timeseries = Experiment(1).timeseries()
    sim_timeseries = Simulation(1).timeseries()

    plot2_SBE(exp_timeseries)
    plot2_SBE(sim_timeseries, pos_y=1, labels=('c','d'))

    scal = defaultdict(dict)
    for c in FIGURE_CULTURES:
        exp_timeseries = Experiment(c).timeseries()
        sim_timeseries = Simulation(c).timeseries()

        scal['original'][c] = burst_measures(exp_timeseries)
        scal['simulated'][c] = burst_measures(sim_timeseries)

    # flatten dictionary with each level stored in key
    df = pd.DataFrame(flatten(scal, ['type', 'culture', 'measure', 'value']))
    df = df.pivot_table(index=['type', 'culture'], columns='measure', values='value')


    plot_scalar(plt.subplot(256), df, 'burst_length')
    plt.ylabel(r'$\mathsf{L_{SBE} [s]}}$', fontsize=16)
    plt.ylim((0,0.6))
    plt.title('e', loc='left', fontsize=18)

    plot_scalar(plt.subplot(257), df, 'interburst_length')
    plt.ylabel(r'$\mathsf{L_{IBI} [s]}}$', fontsize=16)
    plt.title('f', loc='left', fontsize=18)

    plot_scalar(plt.subplot(258), df, 'firing_rate')
    plt.ylabel(r'$\mathsf{f\ [Hz]}}$', fontsize=16)
    plt.title('g', loc='left', fontsize=18)

    plot_scalar(plt.subplot(259), df, 'burst_recruitment')
    plt.ylabel(r'$\mathsf{neuron/burst\ [\%]}}$', fontsize=16)
    plt.ylim((0,100))
    plt.title('h', loc='left', fontsize=18)

    ax = plt.subplot2grid((2, 5), (1, 4), colspan=1)
    plt.scatter(None, None, 20, color='gray', marker='s', label='experiment')
    plt.scatter(None, None, 20, color='brown', marker='s', label='simulation')
    plt.axis('off')
    plt.legend(loc='lower center', scatterpoints=1, markerscale=3., frameon=False)

    show_or_savefig(figpath, figurename)


def test_spont():

    timeseries = Experiment(1).timeseries()

    weighted_bursts, analysed_timeseries, _, _ = detect_SBE(timeseries, t_interval=120)

    intervals = np.array(weighted_bursts[['begin', 'end']].sort_values('begin').values)

    # spontaneous active neurons = neurons that are active in inter burst intervals (IBI)
    list_periods = defaultdict(list)
    n_IBI = len(intervals)
    for IBI_start, IBI_end in zip(intervals[:-1,1], intervals[1:,0]):
        burst_margin = 0.100  # s
        IBI_timeseries = cut_timeseries(analysed_timeseries, t_start=IBI_start + burst_margin,
                                        t_interval=IBI_end - IBI_start - 2 * burst_margin)
        for neuron in IBI_timeseries:
            events = IBI_timeseries[neuron]
            print events
            period = np.median(np.diff(np.sort(events)))
            print IBI_start, IBI_end, neuron, len(events), period

            if not np.isnan(period):
                list_periods[neuron].append(period)

    spontaneous_activity = pd.DataFrame(((neuron, len(periods)/n_IBI, np.nanmedian(periods)) for neuron, periods in list_periods.items()), columns=['neuron', 'activity', 'period'])

    print spontaneous_activity

    plt.plot(spontaneous_activity['activity'], spontaneous_activity['period'], '.')
    plt.show()


    raster_plot(plt.subplot(111), analysed_timeseries, use='neuron')
    plt.show()


def burst_measures(timeseries):
    weighted_bursts, analysed_timeseries, _, _ = detect_SBE(timeseries, t_interval=120)

    intervals = np.array(weighted_bursts[['begin', 'end']].sort_values('begin').values)

    burst_length = np.median(np.diff(intervals, axis=1))

    interburst_length = np.median(np.diff(np.vstack((intervals[1:,0], intervals[:-1,1]))))

    # number of neurons active in burst
    burst_recruitment = 10
    n_neurons = len(analysed_timeseries.keys())
    br = list()
    for begin, end in intervals:
        active_neurons=0
        for neuron, ts in analysed_timeseries.items():
            if np.any(np.logical_and(ts>begin, ts<end)):
                active_neurons +=1
        br.append(active_neurons/n_neurons * 100)   # in percent
    burst_recruitment = np.nanmedian(br)

    # median of median firing rate of each neurons
    fr = list()
    for neuron in analysed_timeseries:
        period = np.median(np.diff(np.sort(analysed_timeseries[neuron])))
        fr.append(1 / period)
    firing_rate = np.nanmedian(fr)


    return {'burst_length' : float(burst_length),
            'burst_recruitment': float(burst_recruitment),
            'interburst_length' : float(interburst_length),
            'firing_rate': float(firing_rate),
            }


def plot_scalar(ax, df, scalar_name):
    xorder = ['original', 'simulated']
    scalar = df[scalar_name].unstack(level=-1).reindex(xorder)
    logging.info('scalar %s:' % scalar_name)
    logging.info(scalar)

    print_test_result(scalar_name, np.array(scalar).T, )

    bplot = ax.boxplot(np.array(scalar).T,
                       boxprops={'color': "black", "linestyle": "-"}, patch_artist=True, widths=0.7)

    colors = ['gray', 'brown']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_color(color)
    plt.setp(bplot['medians'], color='black')
    plt.setp(bplot['whiskers'], color='black')
    plt.xlabel('network type')
    # ax.set_xticks([1, 2])
    # ax.set_xticklabels(xorder)
    adjust_position(ax,xshrink=0.02, yshrink=0.02)
    without_spines_and_ticks(ax)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off'  # labels along the bottom edge are off
    )


def plot2_SBE(exp_timeseries, pos_y = 0, labels=('a','b')):
    ax1 = plt.subplot2grid((4, 3), (pos_y, 0), colspan=2)
    plotted_exp_timeseries, xlim = plot_SBE(ax1, exp_timeseries)
    plt.title(labels[0], loc='left', fontsize=18)

    ax2 = plt.subplot2grid((4,4), (pos_y, 3), colspan=1)
    raster_plot(ax2, plotted_exp_timeseries)
    ax2.set_xlim(xlim)
    adjust_position(ax2, yshrink=0.02)
    plt.title(labels[1], loc='left', fontsize=18)


def make_supplemental_figure(figurename, figpath=None):
    """Quick plot to show SBE for experiments and simulations """
    for index, datatype in enumerate(( 'experiment', 'simulation' )):

        logging.info('Making figure')
        longfigurename = figurename + '-%d' % (index+1)
        fig = plt.figure(longfigurename, figsize=(20, 15))
        fig.suptitle(datatype, fontsize=14, fontweight='bold')

        plt.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.05)

        for c in FIGURE_CULTURES:
            timeseries = Experiment(c).timeseries() if datatype == 'experiment' else Simulation(c).timeseries()
            ax = plt.subplot(610+c)
            logging.info(c)
            plot_SBE(ax, timeseries, t_interval = 60, smooth_win=10, s=2, gamma=0.1)
            plt.title('culture %d' % c, loc='left', fontsize=18)

        show_or_savefig(figpath, longfigurename)


def cut_timeseries(timeseries, t_start='min', t_interval=10):
    """
    Crop a timeseries to given interval (length).
    :param timeseries: timeseries of (spiking) events indexed by neuron id
    :param t_start: begin of the interval in s (default 'min' = start from the first event)
    :param t_interval: interval length in seconds (default: 10s)
    :return: cropped timeseries (starting a 0s)
    """
    if t_start=='min':
        t_start = min((min(ts) for ts in timeseries.values()))
    t_end = t_start + t_interval  # use 10 s
    cropped_timeseries = {neuron: ts[ts < t_end] - t_start for neuron, ts in timeseries.items()}
    cropped_timeseries = {neuron: ts[ts > 0]  for neuron, ts in cropped_timeseries.items()}

    return cropped_timeseries


def plot_SBE(ax1, timeseries, t_start='min', t_interval = 10, bin_width = 10, smooth_win=10, s=2, gamma=0.25):
    """

    :param ax1:
    :param plotted_timeseries: events indexed by neuron
    :param t_interval: interval length in s (default = 10s)
    :param bin_width: bin width in s (default = 10ms)
    :return:
    """
    weighted_bursts, plotted_timeseries, edges, n_active = detect_SBE(timeseries, t_start, t_interval, bin_width, smooth_win=smooth_win, s=s, gamma=gamma)

    # plotting
    raster_plot(ax1, plotted_timeseries)
    burst_plot(ax1, weighted_bursts)
    adjust_position(ax1, yshrink=0.02)
    ax2 = ax1.twinx()
    ax2.step(edges, n_active, alpha=0.5, where='mid')
    ax2.set_ylabel('active [1/%d ms]' % bin_width)
    adjust_position(ax2, yshrink=0.02)

    try:
        xlim = weighted_bursts[['begin', 'end']].values[0]
    except:
        xlim = None

    return plotted_timeseries, xlim


def burst_plot(ax, weighted_bursts):
    """Plot weighted burst as light red boxes in the background"""
    matrix = weighted_bursts[['begin', 'end']].values
    for begin, end in matrix:
        # plt.hlines(+1, edges[begin - 1], edges[end - 1], 'red', alpha=1.0)  # bar
        ax.add_patch(patches.Rectangle((begin, 0), end - begin, 60, color='red', alpha=0.1))  # box


def detect_SBE(timeseries, t_start='min', t_interval=10, bin_width=10, smooth_win=10, s=2, gamma=0.1):
    """
    Detect synchronised bursting events using Kleinberg's burst detection analysis on batched data.
    :param timeseries: number of active neurons per bin
    :param t_start: starting point of analysis (default: 'min' = start with first event)
    :param t_interval: analysed interval in s (default: 10s)
    :param bin_width: width of time bins for events in ms (default: 10 ms)
    :param smooth_win: width of smooting window in bins (not ms!)
    :param s: multiplicand for the burst levels (see Kleinberg)
    :param gamma: penalty on changing the burst levels (see Kleinberg)
    :return: weighted_bursts: Pandas DataFrame of weighted bursts with columns:
        label: 'SBE'
        begin, end: begin and end of a burst in s
        weight: weight of a burst (likelines of the burst)
    :return: analysed_timeseries: analysed part of the timeseries indexed by neurons
    :return: edges, n_active: used for plotting the number of active neurons
    """

    # using part of the timeseries
    analysed_timeseries = cut_timeseries(timeseries, t_start=t_start, t_interval=t_interval)

    # count events in bins, return number of active neurons at each time bin = n_active
    num_bins = int(t_interval / bin_width * 1000)  # events at seconds, bin_width in ms
    counts = {neuron: np.histogram(ts, bins=num_bins, range=(0, t_interval))[0] for neuron, ts in
              analysed_timeseries.items()}
    active = {neuron: cnts > 0 for neuron, cnts in counts.items()}
    n_active = np.sum(np.vstack(active.values()), axis=0)

    # number of neurons, number of bins, dummy variable with all neurons active all the time = all_active
    n_neurons = len(analysed_timeseries.keys())
    n_bins = len(n_active)
    all_active = n_neurons * np.ones_like(n_active)

    # smoothing by moving average  TODO maybe better to use a binomial kernel
    n_active = np.convolve(n_active.copy(), np.ones((smooth_win,)) / smooth_win, mode='same').astype(int)

    # find the optimal state sequence (q)
    q, all_active, n_active, p = bd.burst_detection(n_active, all_active, n_bins, s=s, gamma=gamma, smooth_win=1)

    # enumerate bursts based on the optimal state sequence
    bursts = bd.enumerate_bursts(q, 'SBE')

    # find weight of bursts
    weighted_bursts = bd.burst_weights(bursts, n_active, all_active, p)

    # burst begin and end: bin index --> seconds
    weighted_bursts.begin = (weighted_bursts.begin-1)*bin_width/1000
    weighted_bursts.end = (weighted_bursts.end-1)*bin_width/1000

    # report
    # logging.info('weighted bursts:')
    # logging.info(weighted_bursts)

    # calculate edges of bins for plotting the number of active neurons
    edges = np.linspace(0, t_interval, num_bins)

    return weighted_bursts, analysed_timeseries, edges, n_active


def raster_plot(ax, timeseries, use='neuron'):
    """Plot events as dots"""
    for index, neuron in enumerate(timeseries):
        t = timeseries[neuron]
        if use=='index':
            ax.plot(t, index * np.ones_like(t), 'k.')
            plt.ylabel('index')
        elif use=='neuron':
            ax.plot(t, neuron * np.ones_like(t), 'k.')
            plt.ylabel('neuron')
    # ax.set_xlim((0,0.5))
    plt.xlabel(r'$t$ [s]')
    # without_spines_and_ticks(ax)


if __name__ == "__main__":
    # test_bd()
    # test_spont()
    make_figure(os.path.basename(__file__))
    make_supplemental_figure(os.path.basename(__file__))
