import os
import numpy as np
from matplotlib import pyplot as plt
import logging

from hana.plotting import annotate_x_bar, mea_axes
from hana.recording import DELAY_EPSILON, neighborhood, electrode_neighborhoods
from hana.segmentation import segment_axon_verbose, restrict_to_compartment

from data import FIGURE_CULTURE, FIGURE_NEURON
from experiment import AxonExperiment
from plotting import show_or_savefig, without_spines_and_ticks, cross_hair, \
    legend_without_multiple_labels, plot_traces_and_delays, adjust_position

logging.basicConfig(level=logging.DEBUG)


def make_figure(figurename, figpath=None):

    # Load electrode coordinates
    neighbors = electrode_neighborhoods(mea='hidens')

    # Load example data
    V, t, x, y, trigger, neuron = AxonExperiment(FIGURE_CULTURE).traces(FIGURE_NEURON)
    t *= 1000  # convert to ms

    # Verbose axon segmentation function
    delay, mean_delay, std_delay, expected_std_delay, thr, valid_delay, index_AIS, positive_delay, axon \
        = segment_axon_verbose(t, V, neighbors)

    logging.info ('Axonal delays:')
    logging.info (restrict_to_compartment(mean_delay, axon))

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 13))
    if not figpath:
        fig.suptitle(figurename + ' Segmentation of the axon based on negative peak at neighboring electrodes',
                 fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.05)

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
    V_AIS = V[index_AIS]  # Showing AIS trace in different color
    V_axons = V[np.where(axon)]   # Showing AIS trace in different color
    ax1.plot(t, V.T,'-', color='gray', label='all' )
    ax1.plot(t, V_axons.T,'-', color='black', label='axons' )
    ax1.plot(t, V_AIS, 'r-', label='AIS')
    ax1.scatter(delay[index_AIS], -550, marker='^', s=100, edgecolor='None', facecolor='red')
    plt.annotate('', (delay[index_AIS]+2, 150), (delay[index_AIS], 150), arrowprops={'arrowstyle': '->', 'color': 'red', 'shrinkA':  0, 'shrinkB': 0})
    annotate_x_bar(delay[index_AIS]+(0,2), 150, text=r'$\tau > \tau_{AIS}$', arrowstyle='->')
    legend_without_multiple_labels(ax1, loc=4, frameon=False)
    ax1.set_xlim((-4,4))
    ax1.set_ylabel(r'V [$\mu$V]')
    ax1.set_xlabel(r'$\Delta$t [ms]')
    without_spines_and_ticks(ax1)
    # label_subplot(ax1, 'A', xoffset=-0.04, yoffset=-0.015)
    plt.title('a', loc='left', fontsize=18)

    # subplot delay map
    ax2 = plt.subplot(332)
    h1 = ax2.scatter(x, y, c=delay, s=10, marker='o', edgecolor='None', cmap='gray')
    h2 = plt.colorbar(h1)
    h2.set_label(r'$\tau$ [ms]')
    h2.set_ticks(np.linspace(-4, 4, num=9))
    add_AIS_and_example_neighborhoods(ax2, x, y, index_AIS, indices_background, indices_foreground)
    mea_axes(ax2)
    # label_subplot(ax2, 'B', xoffset=-0.015, yoffset=-0.015)
    plt.title('b', loc='left', fontsize=18)

    # subplot histogram of delays
    ax3 = plt.subplot(333)
    ax3.hist(delay, bins=len(t), facecolor='gray', edgecolor='gray', label='measured')
    ax3.scatter(delay[index_AIS], 10, marker='v', s=100, edgecolor='None', facecolor='red', zorder=10)
    ax3.hlines(len(x)/len(t), min(t), max(t), color='k', linestyles='--', label='uniform')
    ax3.legend(loc=2, frameon=False)
    ax3.set_ylim((0,200))
    ax3.set_xlim((min(t),max(t)))
    ax3.set_ylabel(r'count')
    ax3.set_xlabel(r'$\tau$ [ms]')
    without_spines_and_ticks(ax3)
    adjust_position(ax3, xshrink=0.02)
    # label_subplot(ax3, 'C', xoffset=-0.04, yoffset=-0.015)
    plt.title('c', loc='left', fontsize=18)

    # ------------- second row

    # Subplot neighborhood with uncorrelated negative peaks
    ax4 = plt.subplot(637)
    plot_traces_and_delays(ax4, V, t, delay, indices_background, offset=-2, ylim=(-10, 5), color=background_color, label='no axon')
    ax4.text(-3.5, -7.5, r'$s_{\tau}$ = %0.3f ms' % std_delay[index_background_example], color=background_color)
    ax4.set_yticks([-10,-5,0,5])
    legend_without_multiple_labels(ax4, loc=4, frameon=False)
    # label_subplot(ax4, 'D', xoffset=-0.04, yoffset=-0.015)
    plt.title('d', loc='left', fontsize=18)

    # Subplot neighborhood with correlated negative peaks
    ax5 = fig.add_subplot(6, 3, 10)
    plot_traces_and_delays(ax5, V, t, delay, indices_foreground, offset=-20, ylim=(-30, 10), color=foreground_color, label='axon')
    ax5.text(-3.5, -22, r'$s_{\tau}$ = %0.3f ms' % std_delay[index_foreground_example], color=foreground_color)
    ax5.set_yticks([-30,-20,-10,0,10])
    legend_without_multiple_labels(ax5, loc=4, frameon=False)
    # label_subplot(ax5, 'E', xoffset=-0.04, yoffset=-0.015)
    plt.title('e', loc='left', fontsize=18)

    # subplot std_delay map
    ax6 = plt.subplot(335)
    h1 = ax6.scatter(x, y, c=std_delay, s=10, marker='o', edgecolor='None', cmap='gray')
    h2 = plt.colorbar(h1, boundaries=np.linspace(0,4,num=256))
    h2.set_label(r'$s_{\tau}$ [ms]')
    h2.set_ticks(np.arange(0, 4.5, step=0.5))
    add_AIS_and_example_neighborhoods(ax6, x, y, index_AIS, indices_background, indices_foreground)
    mea_axes(ax6)
    # label_subplot(ax6, 'F', xoffset=-0.015, yoffset=-0.01)
    plt.title('f', loc='left', fontsize=18)

    # subplot std_delay histogram
    ax7 = plt.subplot(336)
    ax7.hist(std_delay, bins=np.arange(0, max(delay), step=DELAY_EPSILON), facecolor='gray', edgecolor='gray', label='no axons')
    ax7.hist(std_delay, bins=np.arange(0, thr, step=DELAY_EPSILON), facecolor='k', edgecolor='k', label='axons')
    ax7.scatter(std_delay[index_foreground_example], 25, marker='v', s=100, edgecolor='None', facecolor=foreground_color, zorder=10)
    ax7.scatter(std_delay[index_background_example], 25, marker='v', s=100, edgecolor='None', facecolor=background_color, zorder=10)
    ax7.scatter(expected_std_delay, 25, marker='v', s=100, edgecolor='black', facecolor='gray', zorder=10)
    ax7.text(expected_std_delay, 30, r'$\frac{8}{\sqrt{12}}$ ms',  horizontalalignment='center', verticalalignment='bottom', zorder=10)
    ax7.legend(loc=2, frameon=False)
    ax7.vlines(0, 0, 180, color='k', linestyles=':')
    ax7.set_ylim((0, 600))
    ax7.set_xlim((0, 4))
    ax7.set_ylabel(r'count')
    ax7.set_xlabel(r'$s_{\tau}$ [ms]')
    without_spines_and_ticks(ax7)
    adjust_position(ax7, xshrink=0.02)
    # label_subplot(ax7, 'G', xoffset=-0.04, yoffset=-0.01)
    plt.title('g', loc='left', fontsize=18)

    # ------------- third row

    # plot map of delay greater delay of AIS == positive_delay
    ax8 = plt.subplot(337)
    ax8.scatter(x, y, c=positive_delay, s=10, marker='o', edgecolor='None', cmap='gray_r')
    add_AIS_and_example_neighborhoods(ax8, x, y, index_AIS, indices_background, indices_foreground)
    ax8.text(300, 300, r'$\tau > \tau_{AIS}$', fontsize=18, bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    mea_axes(ax8)
    adjust_position(ax8, yshrink=0.02)
    plt.title('h', loc='left', fontsize=18)

    # plot map of thresholded std_delay
    ax9 = plt.subplot(338)
    ax9.scatter(x, y, c=valid_delay, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax9.text(300, 300, r'$s_{\tau} < s_{min}$', fontsize=18, bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    add_AIS_and_example_neighborhoods(ax9, x, y, index_AIS, indices_background, indices_foreground)
    mea_axes(ax9)
    adjust_position(ax9, yshrink=0.02)
    plt.title('i', loc='left', fontsize=18)

    # plot map of axon
    ax10 = plt.subplot(339)
    ax10.scatter(x, y, c=axon, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax10.text(300, 300, 'axon', fontsize=18, bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    add_AIS_and_example_neighborhoods(ax10, x, y, index_AIS, indices_background, indices_foreground)
    mea_axes(ax10)
    adjust_position(ax10, yshrink=0.02)
    plt.title('j', loc='left', fontsize=18)

    show_or_savefig(figpath, figurename)


def add_AIS_and_example_neighborhoods(ax6, x, y, index_AIS, indicies_background, indicies_foreground):
    cross_hair(ax6, x[index_AIS], y[index_AIS])
    ax6.scatter(x[indicies_background], y[indicies_background], s=10, marker='o', edgecolor='None', facecolor='green')
    ax6.scatter(x[indicies_foreground], y[indicies_foreground], s=10, marker='o', edgecolor='None', facecolor='blue')


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))