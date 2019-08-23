import logging
import os
import numpy as np
from matplotlib import pyplot as plt

from hana.plotting import annotate_x_bar, mea_axes
from hana.recording import half_peak_width, half_peak_domain, electrode_neighborhoods, DELAY_EPSILON
from hana.segmentation import segment_dendrite_verbose

from data import FIGURE_CULTURE, FIGURE_NEURON
from experiment import Experiment
from plotting import show_or_savefig, without_spines_and_ticks, cross_hair, \
    legend_without_multiple_labels, adjust_position

logging.basicConfig(level=logging.DEBUG)


# Final figure 4

def make_figure(figurename, figpath=None):

    # Load electrode coordinates
    neighbors = electrode_neighborhoods(mea='hidens')

    # Load example data
    V, t, x, y, trigger, neuron = Experiment(FIGURE_CULTURE).traces(FIGURE_NEURON)
    t *= 1000  # convert to ms

    # Verbose dendrite segmentation function
    delay, mean_delay, std_delay, expected_std_delay, thr, valid_delay, index_AIS, min_delay, max_delay, \
        return_current_delay, dendrite = segment_dendrite_verbose(t, V, neighbors)

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 10))
    if not figpath:
        fig.suptitle(figurename + ' Segmentation of the dendrite based on positive peak at neighboring electrodes',
                     fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.90, bottom=0.05)

    # Position of the AIS
    x_AIS = x[index_AIS]
    y_AIS = y[index_AIS]

    # subplot original unaligned traces
    ax1 = plt.subplot(231)
    V_AIS = V[index_AIS]  # Showing AIS trace in different color
    V_dendrites = V[np.where(dendrite)]  # Showing dendrite trace in different color
    ax1.plot(t, V.T, '-', color='gray', label='all')
    ax1.plot(t, V_dendrites.T, '-', color='black', label='dendrite')
    ax1.plot(t, V_AIS, 'r-', label='AIS')
    annotate_x_bar(half_peak_domain(t, V_AIS), 150, text=' $|\delta_h$| = %0.3f ms' % half_peak_width(t, V_AIS))
    legend_without_multiple_labels(ax1, loc=4, frameon=False)
    ax1.set_ylim((-600, 200))
    ax1.set_ylabel(r'V [$\mu$V]')
    ax1.set_xlim((-1, 1))
    ax1.set_xlabel(r'$\Delta$t [ms]')
    without_spines_and_ticks(ax1)
    plt.title('a', loc='left', fontsize=18)

    # subplot std_delay map
    ax2 = plt.subplot(232)
    h1 = ax2.scatter(x, y, c=std_delay, s=10, marker='o', edgecolor='None', cmap='gray')
    h2 = plt.colorbar(h1, boundaries=np.linspace(0, 4, num=256))
    h2.set_label(r'$s_{\tau}$ [ms]')
    h2.set_ticks(np.arange(0, 4.5, step=0.5))
    cross_hair(ax2, x_AIS, y_AIS)
    mea_axes(ax2)
    plt.title('b', loc='left', fontsize=18)

    # subplot std_delay histogram
    ax3 = plt.subplot(233)
    ax3.hist(std_delay, bins=np.arange(0, max(delay), step=DELAY_EPSILON), facecolor='gray', edgecolor='gray',
             label='nothing')
    ax3.hist(std_delay, bins=np.arange(0, thr, step=DELAY_EPSILON), facecolor='k', edgecolor='k', label='axon/dendrite')
    ax3.scatter(expected_std_delay, 25, marker='v', s=100, edgecolor='black', facecolor='gray', zorder=10)
    ax3.text(expected_std_delay, 30, r'$\frac{8}{\sqrt{12}}$ ms', horizontalalignment='center',
             verticalalignment='bottom', zorder=10)
    ax3.legend(frameon=False)
    ax3.vlines(0, 0, 180, color='k', linestyles=':')
    ax3.set_ylim((0, 600))
    ax3.set_xlim((0, 4))
    ax3.set_ylabel(r'count')
    ax3.set_xlabel(r'$s_{\tau}$ [ms]')
    without_spines_and_ticks(ax3)
    adjust_position(ax3, xshrink=0.02)
    plt.title('c', loc='left', fontsize=18)

    # -------------- second row

    # plot map of delay within half peak domain == return_current_delay
    ax4 = plt.subplot(234)
    ax4.scatter(x, y, c=return_current_delay, s=10, marker='o', edgecolor='None', cmap='gray_r')
    cross_hair(ax4, x_AIS, y_AIS)
    ax4.text(300, 300, r'$\tau \in \delta_h$', fontsize=18, bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    mea_axes(ax4)
    plt.title('d', loc='left', fontsize=18)

    # plot map of thresholded std_delay
    ax5 = plt.subplot(235)
    ax5.scatter(x, y, c=valid_delay, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax5.text(300, 300, r'$s_{\tau} < s_{min}$', fontsize=18, bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    cross_hair(ax5, x_AIS, y_AIS)
    mea_axes(ax5)
    plt.title('e', loc='left', fontsize=18)

    # plot map of dendrite
    ax6 = plt.subplot(236)
    ax6.scatter(x, y, c=dendrite, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax6.text(300, 300, 'dendrite', fontsize=14, bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    cross_hair(ax6, x_AIS, y_AIS)
    mea_axes(ax6)
    plt.title('f', loc='left', fontsize=18)

    show_or_savefig(figpath, figurename)


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))