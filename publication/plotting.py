from __future__ import division

import logging
import os
import sys

import networkx as nx
import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, pearsonr, ttest_ind, median_test

logging.basicConfig(level=logging.DEBUG)


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


def plot_loglog_fit(ax, y, datamarker = '.k', datalabel='data', xlabel ='rank', ylabel ='size', fit=True, rank_threshold=0):
    """
    Log log plot of size distributions with power law fitted.
    :param ax: axis handle
    :param y: data
    :param title: subplot title
    :param datamarker: data marker
    :param datalabel: data label
    :param xlabel: x axis label
    :param ylabel: y axis label
    :param fit: add fit to the data (default: True)
    :param rank_threshold: discard the largest values from fitting (rank<rank_threshold)
    """

    # sort y data in descending order, define corresponding rank starting from 1 for larges y
    y.sort(reverse=True)
    x = range(1, len(y) + 1)

    if fit:
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
    ax.plot(x, y, datamarker, label=datalabel)
    if fit:
        ax.plot(fitted_x, fitted_y, 'k-', label="fit $y=%.3f x ^{%.3f}$" % (popt[0], popt[1]))
    plt.legend(loc='lower left', frameon=False, numpoints=4)

    # Scale and label axes, add title
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


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


def voltage_color_bar(plot_handle, vmin=-20, vmax=20, vstep=5, label=r'$V$ [$\mu$V]', shrink=1.):
    plot_handle.set_clim(vmin=vmin, vmax=vmax)
    colorbar_handle = plt.colorbar(plot_handle, boundaries=np.linspace(vmin, vmax, num=int(vmax-vmin+1)),
                                   extend='both', extendfrac='auto',
                                   shrink=shrink)
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
            n[index] = nx.number_of_nodes(G)
            k[index] = nx.number_of_edges(G)
            D[index] = float(k[index]) / n[index]
            giant = max(nx.connected_component_subgraphs(G.to_undirected()), key=len)
            C[index] = nx.average_clustering(giant)
            L[index] = nx.average_shortest_path_length(giant)
    return values, n, k, C, L, D


def kernel_density (ax, data, orientation='vertical', xscale='lin', yscale=1, style='k-'):
    x = np.log(data) if xscale=='log' else data
    x = x[np.isfinite(x)]
    density = gaussian_kde(x)
    xs = np.linspace(min(x), max(x), 200)
    density.covariance_factor = lambda: .5
    density._compute_covariance()
    yscale = len(x) / sum(density(xs)) if yscale == 'count' else 1
    xp = np.exp(xs) if xscale=='log' else xs
    if orientation=='vertical':
        ax.plot(xp, density(xs) * yscale, style)
    elif orientation=='horizontal':
        ax.plot(density(xs) * yscale, xp, style)


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


def plot_pairwise_comparison(ax, data, measure, ylim=None, legend=True):
    """
    Compare values obtained by different methods as lines connecting values from same subject (e.g. neurons).
    :param ax: axis handle
    :param data: Pandas dataframe
    :param measure: See keys of dictionary returned by ModelDiscriminator.summary
    :param ylim: Hard limits for y axis (default: None)
    :param legend: plotting a legend (default: False)
    :return:
    """
    pivoted = data.pivot(index='method', columns='subject', values=measure)
    if legend:
        pivoted.plot(ax=ax).legend(loc='center left', ncol=2, bbox_to_anchor=(1, 0.5))
    else:
        pivoted.plot(ax=ax,legend=legend)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel('Method')
    ax.set_ylabel(measure)
    adjust_position(ax, xshrink=0.02)
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off')  # ticks along the top edge are off


def show_or_savefig(path, figure_name, dpi=300):
    if path:
       # plt.savefig(os.path.join(path, figure_name + '.eps'))
        plt.savefig(os.path.join(path, figure_name + '.png'), dpi=dpi)
    else:
        plt.show()


def plot_correlation(ax, data, x='', y='', xlim=None, ylim=None,
                     xscale='linear', yscale='linear', density_scaling='count', report=sys.stdout):
    """
    Plot correlation as scatter plot and marginals for instantaneous and delayed (neuron) pairs and report results.
    :param ax: axis handle
    :param data: pandas.DataFrame with columns pre, post, functional_delay, functional_strength, structural_delay,
    structural_strength, synaptic_delay, delayed, simultaneous
    :param xlim, ylim: scaling of x-and y-axis (default None, derive from data)
    :param xscale, yscale: scaling of x-and y-axis (default 'linear')
    :param density_scaling: scaling for marginal plots see kernel_density (default 'count')
    :param report: File handle for report in YAML style (default is sys.stdout and does just print)
    :return: axScatter: handle of Scatter plot
    """

    # Subsetting and counting the data
    delayed = data[data.delayed]
    simultaneous = data[data.simultaneous]
    n_simultaneous = len(simultaneous)
    n_delayed = len(delayed)

    # getting limits
    if xlim is None: xlim = (min(data[x]), max(data[x]))
    if ylim is None: ylim = (min(data[y]), max(data[y]))

    # new axes
    rect_histx, rect_histy, rect_scatter = axes_to_3_axes(ax)
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # scatter plot
    if n_simultaneous ==0 or n_delayed == 0:
        axScatter.scatter(data[x], data[y], color='black', label='all')
    if n_simultaneous > 0:
        axScatter.scatter(simultaneous[x], simultaneous[y], color='red', label='<1 ms')
    if n_delayed > 0:
        axScatter.scatter(delayed[x], delayed[y], color='green', label='>1 ms')

    # Plot marginal distributions
    kernel_density(axHistx, data[x], xscale=xscale, yscale=density_scaling, style='k-')
    kernel_density(axHistx, simultaneous[x], xscale=xscale, yscale=density_scaling, style='r-')
    kernel_density(axHistx, delayed[x], xscale=xscale, yscale=density_scaling, style='g-')
    kernel_density(axHisty, data[y], xscale=yscale, yscale=density_scaling, style='k-', orientation='horizontal')
    kernel_density(axHisty, simultaneous[y], xscale=yscale, yscale=density_scaling, style='r-', orientation='horizontal')
    kernel_density(axHisty, delayed[y], xscale=yscale, yscale=density_scaling, style='g-', orientation='horizontal')

    # Joint legend by proxies
    plt.sca(ax)
    if n_simultaneous > 0: plt.vlines(0, 0, 0, colors='red', linestyles='-', label='<1 ms')
    if n_delayed > 0: plt.vlines(0, 0, 0, colors='green', linestyles='-', label='>1 ms')
    plt.vlines(0, 0, 0, colors='black', linestyles='-', label='all')
    plt.vlines(0, 0, 0, colors='black', linestyles='--', label='x=y')
    plt.legend(frameon=False, fontsize=12)

    # Plot fit and report
    section = {}
    section['all'] = add_linear_fit(axScatter, data[x], data[y],
                                    xscale, yscale, color='black', label='fit (all)')
    if n_simultaneous > 1:
        section['simulatanous'] = add_linear_fit(axScatter, simultaneous[x],
                                                 simultaneous[y], xscale, yscale,
                                                 color='red', label='fit (<1 ms)')
    if n_delayed > 1:
        section['delayed'] = add_linear_fit(axScatter, delayed[x],
                                            delayed[y], xscale, yscale,
                                            color='green', label='fit (>1 ms)')
    yaml.dump({'overlap_vs_z_max': section}, report)

    # add x=y
    axScatter.plot(xlim, ylim,'k--', label='x=y')

    # Legend for Scatterplots and fits
    plt.sca(axScatter)
    plt.legend(frameon=True, loc=2, ncol=2, scatterpoints=1, fontsize=12)
    plt.sca(ax)

    # set limits
    axScatter.set_xlim(xlim)
    axScatter.set_ylim(ylim)
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    # set scales
    axScatter.set_xscale(xscale)
    axScatter.set_yscale(xscale)
    axHistx.set_xscale(xscale)
    axHisty.set_yscale(yscale)
    # no labels
    nullfmt = NullFormatter()  # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHistx.yaxis.set_major_formatter(nullfmt)
    axHisty.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    return axScatter


def add_linear_fit(axScatter, x, y, xscale, yscale, color='black', label='all'):
    # fitting
    x_data = np.log(x) if xscale == 'log' else x
    y_data = np.log(y) if yscale == 'log' else y
    f = np.poly1d(np.polyfit(x_data, y_data, 1))
    # prediction
    x_fit = np.unique(x)
    y_fit = f(np.log(x_fit)) if xscale == 'log' else f(x_fit)
    if yscale == 'log': y_fit = np.exp(y_fit)
    axScatter.plot(x_fit, y_fit, color=color, label=label)
    # Report persons test
    r, p = pearsonr(x_data, y_data)
    n = len(x)
    section = {'Persons_test' : {'r': float(r), 'p':  float(p), 'n': int(n)}}
    return section


def plot_synapse_delays(ax, data, xlim=None, ylim=None,
                        xscale='log', density_scaling ='count', naxes=3, report=sys.stdout):
    """
    Plot corretion and marginals
    :param ax: axes handle
    :param data: pandas.DataFrame with columns pre, post, functional_delay, functional_strength, structural_delay,
    structural_strength, synaptic_delay, delayed, simultaneous
    :param xlim,  ylim: limits for x and y axis of the scatter plot
    :param xscale: scaling for x axis of the scatter plot (default: 'log')
    :param density_scaling: scaling for density axis of the marginal plots (default: 'count')
    :param naxes: if 3 do plot the marginals (default: 3)
    :param report: file handle (default: sys.stdout just prints)
    """
    # TODO Check correct behaviour for naxes == 2!

    # New axes
    if naxes == 2:
        axScatter=ax
        fig = ax.get_figure()
        divider = make_axes_locatable(axScatter)
        axHisty = divider.new_horizontal(size="50%", pad=0.05)
        fig.add_axes(axHisty)
    if naxes == 3:
        rect_histx, rect_histy, rect_scatter = axes_to_3_axes(ax)
        axScatter = plt.axes(rect_scatter)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)

    # Subsetting the data
    n_total = len(data)
    delayed = data[data.delayed]
    n_delayed = len(delayed)
    simultaneous = data[data.simultaneous]
    n_simultanous = len(simultaneous)

    # scatter plot
    axScatter.scatter(simultaneous.functional_strength, simultaneous.synaptic_delay, color='red',
                      label='<1 ms (%d%%)' % (100.0*n_simultanous/n_total))
    axScatter.scatter(delayed.functional_strength, delayed.synaptic_delay, color='green',
                      label='>1 ms (%d%%)' % (100.0*n_delayed/n_total))
    axScatter.set_xscale(xscale)
    axScatter.legend(frameon=False, scatterpoints=1)
    axScatter.set_xlabel(r'$\mathsf{z_{max}}$', fontsize=14)
    axScatter.set_ylabel(r'$\mathsf{\tau_{synapse}=\tau_{spike}-\tau_{axon}\ [ms]}$', fontsize=14)

    # density plot
    kernel_density(axHisty, data.synaptic_delay, yscale=density_scaling, style='k-', orientation='horizontal')
    if naxes==3:
        # joint legend by proxies
        plt.sca(ax)
        plt.vlines(0, 0, 0, colors='green', linestyles='-', label='>1ms')
        plt.vlines(0, 0, 0, colors='red', linestyles='-', label='<1ms')
        plt.vlines(0, 0, 0, colors='black', linestyles='-', label='all')
        plt.legend(frameon=False, fontsize=12)

        kernel_density(axHistx, data.functional_strength, xscale=xscale, yscale=density_scaling, style='k-',
                       orientation='vertical')
        kernel_density(axHistx, simultaneous.functional_strength, xscale=xscale, yscale=density_scaling, style='r-',
                       orientation='vertical')
        kernel_density(axHistx, delayed.functional_strength, xscale=xscale, yscale=density_scaling, style='g-',
                       orientation='vertical')
        axHistx.set_xscale(xscale)

    section={}
    section['delayed'] = {'median': float(np.median(delayed.functional_strength)),
                          'mean': float(np.mean(delayed.functional_strength)),
                          'n': int(n_delayed),
                          'p': float(n_delayed/n_total)}

    section['simultaneous'] = {'median': float(np.median(simultaneous.functional_strength)),
                               'mean': float(np.mean(simultaneous.functional_strength)),
                               'n': int(n_simultanous),
                               'p': float(n_simultanous/n_total)}
    t, p = ttest_ind(np.log(simultaneous.functional_strength), np.log(delayed.functional_strength))
    section['Students_t_test'] = {'p': float(p), 't': float(t)}
    xhi2, p, med, tbl = median_test (simultaneous.functional_strength, delayed.functional_strength)
    section['Moods_median_test'] = {'xhi2': float(xhi2),
                                    'p': float(p),
                                    'median_difference': float(med) }
    yaml.dump({'synapse_z_max': section}, report)

    # define limits
    max_functional_strength = max(data.functional_strength)
    if xlim is None: xlim = (min(data.functional_strength), max_functional_strength*2)  # leave some room on the left
    if ylim is None: ylim = (min(data.synapse_delay), max(data.synapse_delay))

    # set limits
    axScatter.set_xlim(xlim)
    axScatter.set_ylim(ylim)
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    # add hlines to Scatter
    axScatter.hlines(0, 0, max_functional_strength*2, linestyles='--')
    axScatter.hlines(-1, 0, max_functional_strength*2, linestyles=':')
    axScatter.hlines(+1, 0, max_functional_strength*2, linestyles=':')

    # no labels
    nullfmt = NullFormatter()  # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHistx.yaxis.set_major_formatter(nullfmt)
    axHisty.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)


