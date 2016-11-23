import logging

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist

from hana.matlab import load_traces, load_positions
from hana.plotting import annotate_x_bar, set_axis_hidens
from hana.recording import half_peak_width, peak_peak_width, peak_peak_domain, MAXIMUM_NEIGHBORS, NEIGHBORHOOD_RADIUS, \
    DELAY_EPSILON, neighborhood
from publication.plotting import FIGURE_NEURON_FILE, FIGURE_ELECTRODES_FILE, without_spines_and_ticks, cross_hair, \
    legend_without_multiple_labels, label_subplot, plot_traces_and_delays, shrink_axes

logging.basicConfig(level=logging.DEBUG)


# Testing code

def testing_load_traces():
    V, t, x, y, trigger, neuron = load_traces(FIGURE_NEURON_FILE)
    Vtrigger = V[int(trigger)]

    hpw = half_peak_width(t, Vtrigger)
    print hpw

    ppw = peak_peak_width(t, Vtrigger)
    print ppw


# Previous figure 2

def figure02_original(testing=False):
    plt.figure('Figure 2', figsize=(12, 9))

    pos = load_positions(FIGURE_ELECTRODES_FILE)  # only used for set_axis_hidens
    V, t, x, y, trigger, neuron = load_traces(FIGURE_NEURON_FILE)
    t *= 1000  # convert to ms

    # Electrode with most minimal V corresponding to proximal AIS, get coordinates and recorded voltage trace
    index_AIS = np.unravel_index(np.argmin(V), V.shape)[0]
    x_AIS = x[index_AIS]
    y_AIS = y[index_AIS]
    V_AIS = V[int(index_AIS)]

    if testing:  # matplotlib is slow, plotting all traces takes 30 sec
        V = V[range(index_AIS - 10, index_AIS) + range(index_AIS + 1, index_AIS + 11)]  # 20 traces only
        index_AIS = np.unravel_index(np.argmin(V), V.shape)[0]  # get "new" AIS index because old one is shifted

    # find negative peak
    indicies_min = np.argmin(V, axis=1)
    t_min = t[indicies_min]
    index_ais_neg_peak = indicies_min[index_AIS]

    # align traces
    V_aligned = np.array([np.roll(row, shift) for row, shift in zip(V, index_ais_neg_peak - indicies_min - 1)])
    V_aligned_AIS = V_aligned[index_AIS]

    # subplot original unaligned traces
    ax1 = plt.subplot(221)
    ax1.plot(t, V.T, '-', color='gray', label='unaligned')
    ax1.plot(t, V_AIS, 'k-', label='(proximal) AIS')
    annotate_x_bar(peak_peak_domain(t, V_AIS), min(V_AIS) / 2,
                   text=' $\delta_p$ = %0.3f ms' % peak_peak_width(t, V_AIS))
    legend_without_multiple_labels(ax1, loc=4, frameon=False)
    ax1.set_xlim((-0.5, 4))
    ax1.set_ylabel(r'V [$\mu$V]')
    ax1.set_xlabel(r'$\Delta$t [ms]')
    without_spines_and_ticks(ax1)
    label_subplot(ax1, 'A')

    # subplot delay map
    ax2 = plt.subplot(222)
    h1 = ax2.scatter(x, y, c=t_min, s=10, marker='o', edgecolor='None', cmap='gray')
    h2 = plt.colorbar(h1)
    h2.set_label(r'$\tau$ [ms]')
    cross_hair(ax2, x_AIS, y_AIS)
    set_axis_hidens(ax2, pos)
    label_subplot(ax2, 'B', xoffset=-0.02)

    # subplot aligned traces
    ax3 = plt.subplot(223)
    ax3.plot(t, V_aligned.T, '-', color='gray', label='aligned')
    ax3.plot(t, V_aligned_AIS, 'k-', label='(proximal) AIS')
    legend_without_multiple_labels(ax3, loc=4, frameon=False)
    ax3.set_xlim((-0.5, 4))
    ax3.set_ylabel(r'V [$\mu$V]')
    ax3.set_xlabel(r'$\Delta$t [ms]')
    without_spines_and_ticks(ax3)
    label_subplot(ax3, 'C')

    # subplot histogram of delays
    ax4 = plt.subplot(224)
    ax4.hist(t_min, bins=len(t), facecolor='gray', edgecolor='gray')
    ax4.vlines(0, 0, 180, color='k', linestyles=':')
    ax4.hlines(len(x) / len(t), min(t), max(t), color='k', linestyles='--')
    ax4.set_ylabel(r'count')
    ax4.set_xlabel(r'$\Delta$t [ms]')
    without_spines_and_ticks(ax4)
    label_subplot(ax4, 'D')

    plt.show()


# Final figure 2

def figure02():
    fig = plt.figure('Figure 2', figsize=(18,14))

    # Load electrode coordinates
    pos = load_positions(FIGURE_ELECTRODES_FILE)
    neighbors = electrode_neighborhoods(pos)

    # Load example data
    V, t, x, y, trigger, neuron = load_traces(FIGURE_NEURON_FILE)
    t *= 1000  # convert to ms

    # Verbose axon segmentation function
    delay, mean_delay, std_delay, expected_std_delay, thr, valid_delay, index_AIS, positive_delay, axon \
        = __segment_axon(V, t, neighbors)

    logging.info ('Axonal delays:')
    logging.info (axonal_delay(axon, mean_delay))

    # -------------- Plots

    # Define examples
    background_color = 'green'
    index_background_example = 500
    indices_background = neighborhood(neighbors, index_background_example)
    foreground_color = 'blue'
    index_foreground_example = 8624
    indices_foreground = neighborhood(neighbors, index_foreground_example)

    # -------------- first row

    # subplot original unaligned traces
    ax1 = plt.subplot(331)
    ax1.plot(t, V.T,'-', color='gray', label='all' )
    ax1.plot(t, V[index_AIS], 'r-', label='AIS')
    ax1.scatter(delay[index_AIS], -550, marker='^', s=100, edgecolor='None', facecolor='red')
    # annotate_x_bar(peak_peak_domain(t, V_AIS), min(V_AIS)/2, text=' $\delta_p$ = %0.3f ms' % peak_peak_width(t, V_AIS))
    legend_without_multiple_labels(ax1, loc=4, frameon=False)
    ax1.set_xlim((-4,4))
    ax1.set_ylabel(r'V [$\mu$V]')
    ax1.set_xlabel(r'$\Delta$t [ms]')
    without_spines_and_ticks(ax1)
    label_subplot(ax1, 'A', xoffset=-0.04, yoffset=-0.015)

    # subplot delay map
    ax2 = plt.subplot(332)
    h1 = ax2.scatter(x, y, c=delay, s=10, marker='o', edgecolor='None', cmap='gray')
    h2 = plt.colorbar(h1)
    h2.set_label(r'$\tau$ [ms]')
    h2.set_ticks(np.linspace(-4, 4, num=9))
    add_AIS_and_example_neighborhoods(ax2, x, y, index_AIS, indices_background, indices_foreground)
    set_axis_hidens(ax2, pos)
    label_subplot(ax2, 'B', xoffset=-0.015, yoffset=-0.015)

    # subplot histogram of delays
    ax3 = plt.subplot(333)
    ax3.hist(delay, bins=len(t), facecolor='gray', edgecolor='gray', label='measured')
    ax3.scatter(delay[index_AIS], 10, marker='v', s=100, edgecolor='None', facecolor='red', zorder=10)
    # ax3.vlines(0, 0, 180, color='k', linestyles=':')
    ax3.hlines(len(x)/len(t), min(t), max(t), color='k', linestyles='--', label='uniform')
    ax3.legend(frameon=False)
    ax3.set_ylim((0,200))
    ax3.set_xlim((min(t),max(t)))
    ax3.set_ylabel(r'count')
    ax3.set_xlabel(r'$\tau$ [ms]')
    without_spines_and_ticks(ax3)
    shrink_axes(ax3, xshrink=0.01)
    label_subplot(ax3, 'C', xoffset=-0.04, yoffset=-0.015)

    # ------------- second row

    # Subplot neighborhood with uncorrelated negative peaks
    ax4 = plt.subplot(637)
    plot_traces_and_delays(ax4, V, t, delay, indices_background, offset=-2, ylim=(-10, 5), color=background_color, label='no axon')
    ax4.text(-3.5, -7.5, r'$s_{\tau}$ = %0.3f ms' % std_delay[index_background_example], color=background_color)
    ax4.set_yticks([-10,-5,0,5])
    legend_without_multiple_labels(ax4, loc=4, frameon=False)
    label_subplot(ax4, 'D', xoffset=-0.04, yoffset=-0.015)

    # Subplot neighborhood with correlated negative peaks
    ax5 = fig.add_subplot(6, 3, 10)
    plot_traces_and_delays(ax5, V, t, delay, indices_foreground, offset=-20, ylim=(-30, 10), color=foreground_color, label='axon')
    ax5.text(-3.5, -22, r'$s_{\tau}$ = %0.3f ms' % std_delay[index_foreground_example], color=foreground_color)
    ax5.set_yticks([-30,-20,-10,0,10])
    legend_without_multiple_labels(ax5, loc=4, frameon=False)
    label_subplot(ax5, 'F', xoffset=-0.04, yoffset=-0.015)

    # subplot std_delay map
    ax6 = plt.subplot(335)
    h1 = ax6.scatter(x, y, c=std_delay, s=10, marker='o', edgecolor='None', cmap='gray')
    h2 = plt.colorbar(h1, boundaries=np.linspace(0,4,num=256))
    h2.set_label(r'$s_{\tau}$ [ms]')
    h2.set_ticks(np.arange(0, 4.5, step=0.5))
    add_AIS_and_example_neighborhoods(ax6, x, y, index_AIS, indices_background, indices_foreground)
    set_axis_hidens(ax6, pos)
    label_subplot(ax6, 'F', xoffset=-0.015, yoffset=-0.01)

    # subplot std_delay histogram
    ax7 = plt.subplot(336)
    ax7.hist(std_delay, bins=np.arange(0, max(delay), step=DELAY_EPSILON), facecolor='gray', edgecolor='gray', label='no axons')
    ax7.hist(std_delay, bins=np.arange(0, thr, step=DELAY_EPSILON), facecolor='k', edgecolor='k', label='axons')
    ax7.scatter(std_delay[index_foreground_example], 25, marker='v', s=100, edgecolor='None', facecolor=foreground_color, zorder=10)
    ax7.scatter(std_delay[index_background_example], 25, marker='v', s=100, edgecolor='None', facecolor=background_color, zorder=10)
    ax7.scatter(expected_std_delay, 25, marker='v', s=100, edgecolor='black', facecolor='gray', zorder=10)
    ax7.text(expected_std_delay, 30, r'$\frac{8}{\sqrt{12}}$ ms',  horizontalalignment='center', verticalalignment='bottom', zorder=10)
    ax7.legend(frameon=False)
    ax7.vlines(0, 0, 180, color='k', linestyles=':')
    # ax7.hlines(len(x) / len(t), min(t), max(t), color='k', linestyles='--')
    ax7.set_ylim((0, 600))
    ax7.set_xlim((0, 4))
    ax7.set_ylabel(r'count')
    ax7.set_xlabel(r'$s_{\tau}$ [ms]')
    without_spines_and_ticks(ax7)
    shrink_axes(ax7, xshrink=0.01)
    label_subplot(ax7, 'G', xoffset=-0.04, yoffset=-0.01)

    # ------------- third row

    # plot map of delay greater zero
    ax8 = plt.subplot(337)
    ax8.scatter(x, y, c=positive_delay, s=10, marker='o', edgecolor='None', cmap='gray_r')
    add_AIS_and_example_neighborhoods(ax8, x, y, index_AIS, indices_background, indices_foreground)
    ax8.text(300, 300, r'$\tau > \tau_{AIS}$', bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    set_axis_hidens(ax8, pos)
    label_subplot(ax8, 'H', xoffset=-0.005, yoffset=-0.01)

    # plot map of delay greater zero
    ax9 = plt.subplot(338)
    ax9.scatter(x, y, c=valid_delay, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax9.text(300, 300, r'$s_{\tau} < s_{min}$', bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    add_AIS_and_example_neighborhoods(ax9, x, y, index_AIS, indices_background, indices_foreground)
    set_axis_hidens(ax9, pos)
    label_subplot(ax9, 'I', xoffset=-0.005, yoffset=-0.01)

    # plot map of delay greater zero
    ax10 = plt.subplot(339)
    ax10.scatter(x, y, c=axon, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax10.text(300, 300, 'axon', bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    add_AIS_and_example_neighborhoods(ax10, x, y, index_AIS, indices_background, indices_foreground)
    set_axis_hidens(ax10, pos)
    label_subplot(ax10, 'J', xoffset=-0.005, yoffset=-0.01)

    plt.show()


def __segment_axon(V, t, neighbors):
    """
    Verbose segment axon function for figures.
    :param V:
    :param t:
    :param neighbors:
    :return: all internal variables
    """
    delay = find_peaks(V, t)
    index_AIS = find_AIS(V)
    mean_delay, std_delay = neighborhood_statistics(delay, neighbors)
    expected_std_delay = mean_std_for_random_delays(delay)
    thr = find_valley(std_delay, expected_std_delay)
    valid_delay = std_delay < thr
    positive_delay = mean_delay > delay[index_AIS]
    axon = np.multiply(positive_delay, valid_delay)
    return delay, mean_delay, std_delay, expected_std_delay, thr, valid_delay, index_AIS, positive_delay, axon


def segment_axon(V, t, neighbors):
    """
    Verbose segment axon function for figures.
    :param V:
    :param t:
    :param neighbors:
    :return: all internal variables
    """
    _, mean_delay, _, _, _, _, _, _, axon = __segment_axon(V, t, neighbors)
    delay = axonal_delay(axon, mean_delay)
    return delay


def axonal_delay(axon, mean_delay):
    """
    Return mean_delay for axon, NaN otherwise.
    :param axon: boolean array
    :param mean_delay: array
    :return: delay: array
    """
    delay = mean_delay
    delay[np.where(np.logical_not(axon))] = np.NAN
    return delay


def find_valley(std_delay, expected_std_delay):
    # Find valley between peak for axons and peak for random peak at expected_std_delay
    hist, bin_edges = np.histogram(std_delay, bins=np.arange(0, expected_std_delay, step=DELAY_EPSILON))
    index_thr = np.argmin(hist)
    thr = bin_edges[index_thr + 1]
    return thr


def mean_std_for_random_delays(delay):
    # Calculated expected_std_delay assuming a uniform delay distribution
    expected_std_delay = (max(delay) - min(delay)) / np.sqrt(12)
    return expected_std_delay


def neighborhood_statistics(delay, neighbors):
    # Calculate mean delay, and std_delay
    sum_neighbors = sum(neighbors)
    mean_delay = np.divide(np.dot(delay, neighbors), sum_neighbors)
    diff_delay = delay - mean_delay
    var_delay = np.divide(np.dot(np.power(diff_delay, 2), neighbors), sum_neighbors)
    std_delay = np.sqrt(var_delay)
    return mean_delay, std_delay


def find_AIS(V):
    """
    Electrode with most minimal V corresponding to (proximal) AIS
    :param V: recorded traces
    :return: electrode_index: index of the electrode near to the AIS
    """
    electrode_AIS = np.unravel_index(np.argmin(V), V.shape)[0]
    return electrode_AIS


def find_peaks(V, t, negative_peaks=True):
    """
    Find timing of negative (positive) peaks.
    :param V: matrix containing the trace for each electrode
    :param t: time
    :param negative_peaks: detect negative peaks if true, positive peak otherwise
    :return: delays: delay for each electrode
    """
    indices = np.argmin(V, axis=1) if negative_peaks else np.argmax(V, axis=1)
    delay = t[indices]
    return delay


def electrode_neighborhoods(pos):
    """
    Calculate neighbor matrix from distances between electrodes.
    :param pos: electrode coordinates
    :return: neighbors: square matrix
    """
    pos_as_array = np.asarray(zip(pos.x, pos.y))
    distances = squareform(pdist(pos_as_array, metric='euclidean'))
    neighbors = distances < NEIGHBORHOOD_RADIUS
    sum_neighbors = sum(neighbors)
    assert (max(sum_neighbors)) <= MAXIMUM_NEIGHBORS  # sanity check
    return neighbors


def add_AIS_and_example_neighborhoods(ax6, x, y, index_AIS, indicies_background, indicies_foreground):
    cross_hair(ax6, x[index_AIS], y[index_AIS])
    ax6.scatter(x[indicies_background], y[indicies_background], s=10, marker='o', edgecolor='None', facecolor='green')
    ax6.scatter(x[indicies_foreground], y[indicies_foreground], s=10, marker='o', edgecolor='None', facecolor='blue')


# testing_load_traces()
figure02()
