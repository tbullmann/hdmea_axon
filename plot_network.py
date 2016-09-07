from itertools import chain

import numpy as np
from matplotlib import pyplot as plt


def set_axis_hidens(ax, pos):
    ax.set_aspect('equal')
    margin = 20
    ax.set_xlim([min(pos.x) - margin, max(pos.x) + margin])
    ax.set_ylim([min(pos.y) - margin, max(pos.y) + margin])
    ax.set_xlabel(r'x [$\mu$m]')
    ax.set_ylabel(r'y [$\mu$m]')


def unique_neurons(pair_dict):
    neuron_set = set(list(chain(*list(pair_dict.keys()))))
    return neuron_set


def plot_neuron_id(ax, neuron_dict, pos):
    for neuron in neuron_dict:
        ax.annotate(' %d' % neuron, (pos.x[neuron], pos.y[neuron]))


def plot_neuron_points(ax, neuron_dict, pos):
    for neuron in neuron_dict:
        ax.scatter(pos.x[neuron], pos.y[neuron], s=18, marker='o', color='k')


def plot_network(ax, pair_dict, pos):
    for (pre, post) in pair_dict:
        ax.annotate('', (pos.x[pre], pos.y[pre]), (pos.x[post], pos.y[post]), arrowprops={'arrowstyle': '<-'})

def highlight_connection (ax, neuron_pair, pos, annotation_text=None):
    pre, post = neuron_pair
    ax.annotate('', (pos.x[pre], pos.y[pre]), (pos.x[post], pos.y[post]),
                arrowprops={'arrowstyle': '<-', 'color':'r', 'linewidth':2})
    ax.scatter(pos.x[pre], pos.y[pre], s=18, marker='o', color='r')
    ax.scatter(pos.x[post], pos.y[post], s=18, marker='o', color='r')
    if annotation_text is not None:
        x = np.mean((pos.x[pre],pos.x[post]))
        y = np.mean((pos.y[pre],pos.y[post]))
        plt.text(x, y, annotation_text, color = 'r')


def plot_pair_func(timelags, timeseries_hist, surrogates_mean, surrogates_std, std_score, title):
    ax1 = plt.subplot(211)
    plot_timeseries_hist_and_surrogates(ax1, timelags, timeseries_hist, surrogates_mean, surrogates_std)
    ax1.set_title(title)
    ax2 = plt.subplot(212)
    plot_std_score_and_peaks(ax2, timelags, std_score)


def plot_std_score_and_peaks(axis, timelags, std_score, thr=10, peak=None):
    axis.step(timelags, std_score, label="standard score", color='k', linewidth=1, where='mid')
    axis.set_xlim([np.min(timelags), np.max(timelags)])
    axis.set_xlabel("time lag [s]")
    axis.set_ylabel("(normalized)")
    if thr is not None:
        axis.hlines(thr, np.min(timelags), np.max(timelags), colors='k', linestyles='dashed', label='threshold')
    if peak is not None:
            axis.vlines(peak, thr, np.max(std_score), colors='r', linewidth=4, label='peak')
    axis.legend()


def plot_timeseries_hist_and_surrogates(axis, timelags, timeseries_hist, surrogates_mean, surrogates_std):
    axis.step(timelags, timeseries_hist, label="original histogram", color='k', linewidth=1, where='mid')
    axis.step(timelags, surrogates_mean, label="surrogates (mean)", color='b', linewidth=1, where='mid')
    axis.step(timelags, surrogates_mean - surrogates_std, label="surrogates (std)", color='b', linewidth=1,
              linestyle='dotted', where='mid')
    axis.step(timelags, surrogates_mean + surrogates_std, color='b', linewidth=1, linestyle='dotted', where='mid')
    axis.set_xlim([np.min(timelags), np.max(timelags)])
    axis.set_xlabel("time lag [s]")
    axis.set_ylabel("count")
    axis.legend()