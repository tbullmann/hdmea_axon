from __future__ import division
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

from publication.data import Experiment, FIGURE_CULTURES
from publication.plotting import show_or_savefig, without_spines_and_ticks, adjust_position
from publication.comparison import print_test_result
from publication import burst_detection as bd

import logging
logging.basicConfig(level=logging.DEBUG)



def make_figure(figurename, figpath=None):

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 11))
    if not figpath:
        fig.suptitle(figurename + '    Dynamics of networks', fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.1)

    for c in FIGURE_CULTURES:
        timeseries = Experiment(c).timeseries()
        ax = plt.subplot2grid((6, 2), (c-1, 0))

        logging.info(c)
        plot_SBE(ax, timeseries, t_interval=60, smooth_win=10, s=2, gamma=0.1)
        # ax.axis('off')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_ticks([])
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title('culture %d' % c, loc='center', fontsize=12)

        if c == 1:
            plt.title('a', loc='left', fontsize=18)
        if c == 6:
            plt.xlabel('t [s]')
            max_y = max(ax.get_ylim())
            ax.set_ylim((-2, max_y))
        else:
            ax.get_xaxis().set_ticks([])
            ax.spines['bottom'].set_visible(False)


    BLs = list()
    IBLs = list()
    BRs = list()
    ISIs = list()

    for c in FIGURE_CULTURES:
        exp_timeseries = Experiment(c).timeseries()
        BL, IBL, BR, ISI = burst_measures(exp_timeseries)
        BLs.append(BL.astype(np.float))
        IBLs.append(IBL.astype(np.float))
        BRs.append(BR)
        ISIs.append(ISI)

    ax1 = plt.subplot2grid((2, 4), (0, 2))
    plot_distributions(ax1, BLs, fill_color='magenta') # 0-2s
    ax1.set_xlim((0,2))
    ax1.set_xlabel('burst length [s]')
    adjust_position(ax1, xshrink=0.01)
    plt.title('b', loc='left', fontsize=18)

    ax2 = plt.subplot2grid((2, 4), (0, 3))
    plot_distributions(ax2, IBLs, fill_color='magenta') # 0-30s
    ax2.set_xlim((0,30))
    ax2.set_xlabel('inter burst length [s]')
    adjust_position(ax2, xshrink=0.01)
    plt.title('c', loc='left', fontsize=18)

    ax3 = plt.subplot2grid((2, 4), (1, 2))
    plot_distributions(ax3, BRs, fill_color='magenta') # 0-100%
    ax3.set_xlim((0,100))
    ax3.set_xlabel('burst recruitment [%]')
    adjust_position(ax3, xshrink=0.01)
    plt.title('d', loc='left', fontsize=18)

    ax4 = plt.subplot2grid((2, 4), (1, 3))
    plot_distributions(ax4, ISIs, fill_color='gray') # 0-4s
    ax4.set_xlim((-3,2))
    ax4.set_xticks([-3,-2,-1,0,1,2])
    ax4.set_xticklabels([r'$\mathsf{10^{-3}}$',r'$\mathsf{10^{-2}}$',r'$\mathsf{10^{-1}}$',r'$\mathsf{10^{0}}$',r'$\mathsf{10^{1}}$',r'$\mathsf{10^{2}}$'])
    ax4.set_xticklabels([0.001,0.01,0.1,1,10,100])
    ax4.set_xlabel('inter spike interval [s]')
    adjust_position(ax4, xshrink=0.01)
    plt.title('e', loc='left', fontsize=18)

    show_or_savefig(figpath, figurename)


def plot_distributions(ax, values, fill_color='gray'):
    # show distirbution of median as boxplot
    bplot = plt.boxplot([np.median(x) for x in values], 0, '+', 0, positions=np.array([0.]),
                        boxprops={'color': "black", "linestyle": "-"}, patch_artist=True, widths=0.7)
    bplot['boxes'][0].set_color(fill_color)
    plt.setp(bplot['medians'], color='black')
    plt.setp(bplot['whiskers'], color='black')
    # add violin plot for every culture
    vplot = plt.violinplot(values, FIGURE_CULTURES, points=80, vert=False, widths=0.7,
                           showmeans=False, showextrema=True, showmedians=True)
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
    ax.set_ylabel('culture')
    ax.set_yticks([0, ] + list(FIGURE_CULTURES))
    ax.set_yticklabels(['all', ] + list(FIGURE_CULTURES))
    ax.set_ylim((-0.5, 6.5))
    ax.invert_yaxis()


def burst_measures(timeseries):
    weighted_bursts, analysed_timeseries, _, _ = detect_SBE(timeseries, t_interval=600)

    intervals = np.array(weighted_bursts[['begin', 'end']].sort_values('begin').values)

    BL = np.diff(intervals, axis=1)

    IBL = np.diff(np.vstack((intervals[1:,0], intervals[:-1,1])))

    # number of neurons active in burst
    n_neurons = len(analysed_timeseries.keys())
    BR = list()
    for begin, end in intervals:
        active_neurons=0
        for neuron, ts in analysed_timeseries.items():
            if np.any(np.logical_and(ts>begin, ts<end)):
                active_neurons +=1
        BR.append(active_neurons/n_neurons * 100)   # in percent

    # median of median firing rate of each neurons
    log10ISI = list()
    for neuron in analysed_timeseries:
        period = np.median(np.diff(np.sort(analysed_timeseries[neuron])))
        if np.isfinite(period):
            log10ISI.append(np.log10(period))

    return BL, IBL, BR, log10ISI


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
    # adjust_position(ax1, yshrink=0.01)

    return plotted_timeseries


def burst_plot(ax, weighted_bursts):
    """Plot weighted burst as light red boxes in the background"""
    matrix = weighted_bursts[['begin', 'end']].values
    for begin, end in matrix:
        # plt.hlines(+1, edges[begin - 1], edges[end - 1], 'red', alpha=1.0)  # bar
        ax.add_patch(patches.Rectangle((begin, 0), end - begin, 60, color='magenta'))  # box


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


def raster_plot(ax, timeseries):
    """Plot events as vertical lines"""
    l = len(timeseries)
    markersize = 100.0 / l
    print l, markersize
    for index, neuron in enumerate(timeseries):
        t = timeseries[neuron]
        ax.plot(t, index * np.ones_like(t), 'k|', markersize=markersize)
    ax.set_ylim((0,l))


if __name__ == "__main__":
    # test_bd()
    # test_spont()
    make_figure(os.path.basename(__file__))
    # make_supplemental_figure(os.path.basename(__file__))
