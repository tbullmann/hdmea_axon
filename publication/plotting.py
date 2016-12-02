import logging

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.DEBUG)


FIGURE_ARBORS_MATFILE = 'data/hidens2018at35C_arbors.mat'
FIGURE_EVENTS_FILE = 'data/hidens2018at35C_events.mat'
FIGURE_ELECTRODES_FILE = 'data/hidens_electrodes.mat'
FIGURE_NEURON_FILE = 'data/neuron5.h5'
FIGURE_NEURON_FILE_FORMAT = 'data/neuron%d.h5'
# FIGURE_NEURONS = [2, 3, 4, 5, 10, 11, 13, 17, 20, 21, 22, 23, 25, 27, 29, 31, 35, 36, 37, 41, 49, 50, 51, 59, 62]
FIGURE_NEURONS = [2, 3, 4, 5, 10, 11, 13, 20, 21, 22, 23, 25, 27, 29, 31, 35, 36, 37, 41, 49, 50, 51, 59]

def plot_parameter_dependency(ax, Z, x, y, w=None, levels=None, fmt='%d', legend_loc='lower right'):
    """
    Plotting parameter dependency.
    :param ax: axis handle
    :param Z: dictionary containing values for parameters x, y, w
    :param x: parameter values for x axis
    :param y: parameter values for x axis
    :param w: (two) parameter values for overlay, w=None if not applicable
    :param levels: list of Z values, for which contour lines are plotted
    :param fmt: format string for labeling of contour lines
    :return:
    """
    from scipy.ndimage.filters import gaussian_filter
    import matplotlib.patches as mpatches

    X, Y = np.meshgrid(x, y, indexing='ij')

    if w is None:  # standard black and white blot
        CS1 = ax.contour(X, Y, gaussian_filter(Z, 1), levels=levels, colors='k')
        ax.clabel(CS1, fmt=fmt, inline=1, fontsize=10, inline_spacing=-5)
    else:  # first entry entry in dictionary in black, second in red and dashed lines
        CS1 = ax.contour(X, Y, gaussian_filter(Z[w[0]], 1), levels=levels, colors='k')
        ax.clabel(CS1, fmt=fmt, inline=1, fontsize=10, inline_spacing=-5)
        CS2 = ax.contour(X, Y, gaussian_filter(Z[w[1]], 1), levels=levels, colors='r',
                         linestyles='dashed')
        ax.clabel(CS2, fmt=fmt, inline=1, fontsize=10, inline_spacing=-5)
        # Zinf = np.float64(np.isfinite(Z[w[0]]))
        # CS1 = ax.contourf(X, Y, gaussian_filter(Zinf, 1), levels=(0,1), colors='y')
        black_patch = mpatches.Patch(color='black', label=w[0])
        red_patch = mpatches.Patch(color='red', label=w[1])
        plt.legend(loc=legend_loc, handles=[black_patch, red_patch])


def plot_loglog_fit(ax, y, title = 'size distribution', datalabel = 'data', xlabel = 'rank', ylabel = 'size'):
    """
    Log log plot of size distributions with power law fitted.
    :param ax: axis handle
    :param y: data
    :param title: subplot title
    :param datalabel: data label
    :param xlabel: x axis label
    :param ylabel: y axis label
    """

    # sort y data in descending order, define corresponding rank starting from 1 for larges y
    y.sort(reverse=True)
    x = range(1, len(y) + 1)

    # Define domain for the fitted curves
    min_exp = 0
    max_exp = np.ceil(np.log10(len(y)))
    fitted_x = np.logspace(min_exp, max_exp, base=10)

    # Define fitted curve
    def f(x, a, b): return a * np.power(x, b)

    # Fit curve to data and predict y
    popt, pcov = curve_fit(f, x, y)
    fitted_y = f(fitted_x, *popt)

    # Plot data and legend
    ax.plot(x, y, '.k', label=datalabel)
    ax.plot(fitted_x, fitted_y, 'r-', label="fit $y=%.3f x ^{%.3f}$" % (popt[0], popt[1]))
    plt.legend(loc='upper right')

    # Scale and label axes, add title
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def without_spines_and_ticks(ax):
    """Remove spines and ticks on left and bottom side of the plot"""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def cross_hair(ax2, x, y, color='red'):
    """Plot a simple crosshair"""
    ax2.plot(x, y, marker='$\\bigoplus$', markersize=20, color=color)


def legend_without_multiple_labels(ax, **kwargs):
    """Stop matplotlib repeating labels in legend"""
    from collections import OrderedDict
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), **kwargs)


def label_subplot(ax, text, xoffset=-0.06, yoffset=0):
    """Labelthe subplot in the upper left corner."""
    position = ax.get_position()
    logging.info('Subplot %s with position:' % text)
    logging.info(position)
    fig = ax.figure
    fig.text(position.x0 + xoffset, position.y1+yoffset, text, size=30, weight='bold')


def plot_traces_and_delays(ax, V, t, delay, indicies, offset=None, ylim=None, color='k', label=None):
    """
    Plot traces and delays (of a neighborhod of electrodes).
    :param ax: axis handle
    :param V: (all) Voltage traces
    :param t: time
    :param delay: (all) delays
    :param indicies: traces indicicating what part of V and delay should be plotted
    :param offset: y offset for the markers
    """
    if not offset: offset = max(V)
    ax.plot(t, V[indicies].T, '-', color=color, label=label)
    ax.scatter(delay[indicies], offset * np.ones(7), marker='^', s=100, edgecolor='None', facecolor=color)
    ax.set_xlim((min(t), max(t)))
    if ylim: ax.set_ylim(ylim)
    ax.set_ylabel(r'V [$\mu$V]')
    ax.set_xlabel(r'$\Delta$t [ms]')
    without_spines_and_ticks(ax)
    shrink_axes(ax, yshrink = 0.01)


def shrink_axes(ax, yshrink = 0, xshrink = 0):
    """
    Shrink height to fit with surrounding plots.
    :param ax: axis handle
    :param yshrink: shrinkage in figure coordinates (0..1)
    """
    position = ax.get_position()
    position.y0 += yshrink
    position.y1 -= yshrink
    position.x0 += xshrink
    position.x1 -= xshrink
    ax.set_position(position)


FIGURE_ARBORS_FILE = 'temp/all_neurites.h5'