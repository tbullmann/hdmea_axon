import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

import os, pickle, logging
logging.basicConfig(level=logging.DEBUG)


FIGURE_ARBORS_FILE = 'data/hidens2018at35C_arbors.mat'
FIGURE_EVENTS_FILE = 'data/hidens2018at35C_events.mat'
FIGURE_ELECTRODES_FILE = 'data/hidens_electrodes.mat'


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


def plot_pcg(ax, polychronous_group, color='r'):
    """
    Plots a polychronous group with colored arrows. Unfortunately, (re)plotting is slow for large groups.
    :param ax: axis handle
    :param polychronous_group: graph containing the events and connections of a polychronous group
    :param color: color of arrows, default is red
    :return:
    """
    times, neurons = (zip(*polychronous_group.nodes()))
    ax.plot(times, neurons, '.k', markersize=10)
    for ((time1, neuron1), (time2, neuron2)) in polychronous_group.edges():
        if time1 < time2: time1, neuron1, time2, neuron2 = time2, neuron2, time1, neuron1
        ax.annotate('', (time1, neuron1), (time2, neuron2), arrowprops={'color': color, 'arrowstyle': '->'})
    # Annotations alone do not resize the plotting windows, do:
    # ax.set_xlim([min(times), max(times)])
    # ax.set_ylim([min(neurons), max(neurons)])
    plt.xlabel("time [s]")
    plt.ylabel("neuron index")


def plot_pcgs(ax, list_of_polychronous_groups):
    """
    Plots each polychronous group with different (arrow) colors. (Re)Plotting is painfully slow.
    :param ax: axis handle
    :param list_of_polychronous_groups: list of polychronous groups
    """
    from itertools import cycle
    cycol = cycle('bgrcmk').next
    for g in list_of_polychronous_groups:
        plot_pcg(ax, g, color=cycol())

