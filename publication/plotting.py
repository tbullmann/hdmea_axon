import logging

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

logging.basicConfig(level=logging.DEBUG)


FIGURE_EVENTS_FILE = 'data/events.h5'
FIGURE_ARBORS_FILE = 'temp/all_neurites.h5'
FIGURE_NEURON_FILE = 'data/neuron5.h5'
FIGURE_NEURON_FILE_FORMAT = 'data/neuron%d.h5'
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
    :param legend_loc: Placement of legend
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


def plot_loglog_fit(ax, y, title = 'size distribution', datalabel = 'data', xlabel = 'rank', ylabel = 'size', rank_threshold=0):
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
    popt, pcov = curve_fit(f, x[rank_threshold:], y[rank_threshold:])
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
    adjust_position(ax, yshrink = 0.01)


def adjust_position(ax, yshrink=0, xshrink=0, xshift=0, yshift=0):
    """
    Adjust position of bounding box for a (sub)plot.
    :param ax: axis handle
    :param xshrink, yshrink: shrinkage in figure coordinates (0..1)
    :param xshift, yshift: shift in figure coordinates (0..1)
    """
    position = ax.get_position()
    position.y0 += yshift + yshrink
    position.y1 += yshift - yshrink
    position.x0 += xshift + xshrink
    position.x1 += xshift - xshrink
    ax.set_position(position)


def voltage_color_bar(plot_handle, vmin=-20, vmax=20, vstep=5, label=r'$V$ [$\mu$V]'):
    plot_handle.set_clim(vmin=vmin, vmax=vmax)
    colorbar_handle = plt.colorbar(plot_handle, boundaries=np.linspace(vmin, vmax, num=int(vmax-vmin+1)), extend='both', extendfrac='auto', )
    colorbar_handle.set_label(label)
    colorbar_handle.set_ticks(np.arange(vmin, vmax+vstep, step=vstep))


TEMPORARY_PICKELED_NETWORKS = 'temp/two_networks.p'


def compare_networks(struc, func, nbins=50, scale='linear'):
    """

    :param struc: containing pairs of neuron and some weigth (overlap or score)
    :param nbins: if none, than all unique values will be used as threshold, otherwise linspace with nbins
    :return:
    """

    structural_thresholds = __get_marginals(struc, nbins=nbins, scale=scale)
    functional_thresholds = __get_marginals(func, nbins=nbins, scale=scale)

    result_shape = (len(structural_thresholds), len(functional_thresholds))

    intersection_size = np.zeros(result_shape)
    structural_index = np.zeros(result_shape)
    jaccard_index = np.zeros(result_shape)
    functional_index = np.zeros(result_shape)

    for i, structural_threshold in enumerate(structural_thresholds):
        structural_edges = set(key for key, value in struc.items() if value >= structural_threshold)

        for j, functional_threshold in enumerate(functional_thresholds):
            functional_edges = set(key for key, value in func.items() if value >= functional_threshold)

            intersection_edges = structural_edges & functional_edges
            union_edges = structural_edges | functional_edges

            intersection_size[i,j] = len(intersection_edges)
            if len(union_edges)>0:
                jaccard_index[i,j] = np.float(len(intersection_edges)) / len (union_edges)
            if len(structural_edges)>0:
                structural_index[i,j] = np.float(len(intersection_edges)) / len (structural_edges)
            if len(functional_edges)>0:
                functional_index[i,j] = np.float(len(intersection_edges)) / len (functional_edges)

    return structural_thresholds, functional_thresholds, \
           intersection_size, structural_index, jaccard_index, functional_index


def format_parameter_plot(ax):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\mathsf{\rho\ [\mu m^2]}$', fontsize=14)
    ax.set_ylabel(r'$\mathsf{\zeta}$', fontsize=14)


def __get_marginals(dictionary, nbins, scale='linear'):

    if nbins:
        min_value = min(dictionary.values())
        max_value = max(dictionary.values())
        if scale == 'linear':
            values = np.linspace(min_value, max_value, nbins)
        elif scale == 'log':
            max_exp = np.log(max_value / min_value)
            exponents = np.linspace(1, max_exp, nbins)
            values = min_value * np.exp(exponents)
    else:
        values = np.unique(dictionary.values())

    return values


def analyse_networks(dictionary, nbins=None):
    """

    :param dictionary: containing pairs of neuron and some weigth (overlap or score)
    :param nbins: if none, than all unique values will be used as threshold, otherwise linspace with nbins
    :return:
    """

    values = __get_marginals(dictionary, nbins)

    G = nx.DiGraph()

    n_values = len(values)
    n = np.zeros(n_values)
    k = np.zeros(n_values)
    C = np.zeros(n_values)
    L = np.zeros(n_values)
    D = np.zeros(n_values)

    for index, threshold in enumerate(values):
        edges = [key for key, value in dictionary.items() if value >= threshold]
        G.clear()
        G.add_edges_from(edges)
        if G:
            giant = max(nx.connected_component_subgraphs(G.to_undirected()), key=len)
            n[index] = nx.number_of_nodes(giant)
            k[index] = nx.number_of_edges(giant)
            C[index] = nx.average_clustering(giant)
            L[index] = nx.average_shortest_path_length(giant)
            D[index] = float(k[index]) / n[index]
    return values, n, k, C, L, D


def correlate_two_dicts(xdict, ydict, subset_keys=None):
    """Find values with the same key in both dictionary and return two arrays of corresponding values"""
    x, y, _ = __correlate_two_dicts(xdict, ydict, subset_keys)
    return x, y


def __correlate_two_dicts(xdict, ydict, subset_keys=None):
    x, y, keys = [], [], []
    both = set(xdict.keys()) & set(ydict.keys())
    if subset_keys: both = both & set(subset_keys)
    for pair in both:
        x.append(xdict[pair])
        y.append(ydict[pair])
        keys.append(pair)
    x = np.array(x)
    y = np.array(y)
    return x, y, keys


def kernel_density (ax, data, orientation='vertical', scaling=1, style='k-'):
    if scaling == 'count':
        scaling = len(data)
    density = gaussian_kde(data)
    xs = np.linspace(min(data), max(data), 200)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    if orientation=='vertical':
        ax.plot(xs, density(xs) * scaling, style)
    elif orientation=='horizontal':
        ax.plot(density(xs) * scaling, xs, style)


def axes_to_3_axes(ax, factor = 0.75, spacing = 0.01):
    """Convert axis to one for the scatter and two adjacent histograms for marginal distributions."""
    position = ax.get_position()
    ax.axis('off')
    left, width = position.x0, (position.x1 - position.x0) * factor - spacing / 2
    bottom, height = position.y0, (position.y1 - position.y0) * factor - spacing / 2
    bottom_h, height_h = bottom + height + spacing, (position.y1 - position.y0) * (1 - factor) - spacing / 2
    left_h, width_h = left + width + spacing, (position.x1 - position.x0) * (1 - factor) - spacing / 2
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, height_h]
    rect_histy = [left_h, bottom, width_h, height]
    return rect_histx, rect_histy, rect_scatter