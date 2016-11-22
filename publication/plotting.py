import logging

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.DEBUG)


FIGURE_ARBORS_FILE = 'data/hidens2018at35C_arbors.mat'
FIGURE_EVENTS_FILE = 'data/hidens2018at35C_events.mat'
FIGURE_ELECTRODES_FILE = 'data/hidens_electrodes.mat'
FIGURE_NEURON_FILE = 'data/neuron11.h5'
# Made by the following Matlab code:
# neurons=events.neurons;
# NeuronTable = events.NeuronTable;
# clear events;
# neuron_index = 11;
# x = neurons{neuron_index}.x;
# y = neurons{neuron_index}.y;
# distances = distn([x' y']',[x' y']');
# V = neurons{neuron_index}.mean ;trigger_el_idx = Neuron2Electrode ( NeuronTable, neuron_index);
# pre = -80;
# post = 80;
# sampling_frequency = 20000;
# time = (-pre:post)/sampling_frequency;
# filename = '/Users/tbullmann/Desktop/neuron11.h5'
# hdf5write(filename, '/V',V, '/time',time/1000, '/x',x, '/y',y , '/trigger',trigger_el_idx-1, '/neuron',neuron_index)
# TODO: Fix the BIG BUG in the matlab export script / Neuron2Electrode: Indicies for electrodes in events correctly start at 0, in delays and positive peak the (trigger) electrodes from 1 (!)


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


