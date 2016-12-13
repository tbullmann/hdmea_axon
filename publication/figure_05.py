import logging

import numpy as np
from matplotlib import pyplot as plt

from hana.plotting import set_axis_hidens
from hana.recording import electrode_neighborhoods, load_traces
from hana.segmentation import __segment_dendrite, __segment_axon
from publication.plotting import FIGURE_NEURON_FILE, cross_hair, label_subplot, voltage_color_bar

logging.basicConfig(level=logging.DEBUG)


# Final figure 5

def figure05():

    # Load electrode coordinates
    neighbors = electrode_neighborhoods(mea='hidens')

    # Load example data
    V, t, x, y, trigger, neuron = load_traces(FIGURE_NEURON_FILE)
    t *= 1000  # convert to ms

    # Minimum Voltage and verbose axon segmentation function and
    min_V = np.min(V, axis=1)
    _, _, std_delay_negative_peak, _, thr_std_delay_negative_peak, _, index_AIS, _, axon \
        = __segment_axon(t, V, neighbors)

    # Maximum Voltage and verbose dendrite segmentation function and
    max_V = np.max(V, axis=1)
    _, _, std_delay_positive_peak, _, thr_std_delay_positive_peak, _, _, _, _, _, dendrite \
        =__segment_dendrite(t, V, neighbors)

    # Making figure
    fig = plt.figure('Figure 5', figsize=(18,9))
    fig.suptitle('Figure 5. Using delay distribution at neighboring electrodes is superior to simple peak'
                 ' thresholding in detection of axonal and dendritic signals', fontsize=14, fontweight='bold')

    # Position of the AIS
    x_AIS = x[index_AIS]
    y_AIS = y[index_AIS]

    # -------- first row

    # Map negative peak of voltage
    ax1 = plt.subplot(231)
    ax1_h1 = ax1.scatter(x, y, c=min_V, s=10, marker='o', edgecolor='None', cmap='seismic')
    ax1.text(300, 300, 'negative peak', bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    voltage_color_bar(ax1_h1)
    cross_hair(ax1, x_AIS, y_AIS, color='white')
    set_axis_hidens(ax1)
    label_subplot(ax1, 'A', xoffset=-0.005, yoffset=-0.01)

    # Map of axon
    ax2 = plt.subplot(232)
    ax2.scatter(x, y, c=axon, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax2.text(300, 300, 'axon', bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    cross_hair(ax2, x_AIS, y_AIS, color='white')
    set_axis_hidens(ax2)
    label_subplot(ax2, 'B', xoffset=-0.005, yoffset=-0.01)

    # Std_delay vs max V for dendrite..
    ax3 = plt.subplot(233)
    ax3.scatter(std_delay_negative_peak, min_V, color='gray', marker='o', edgecolor='none', label='all')
    ax3.scatter(std_delay_negative_peak[np.where(axon)], min_V[np.where(axon)], color='blue', marker='^',
                edgecolor='none', label='axon')
    ax3.scatter(std_delay_negative_peak[np.where(dendrite)], min_V[np.where(dendrite)], color='red', marker='d',
                edgecolor='none', label='dendrite')
    ax3.legend(loc=4, frameon=False)
    ax3.text(0.25, -45, 'negative peak', bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    ax3.set_ylim((-50, 0))
    ax3.set_ylabel(r'min $V$ [$\mu$V]')
    ax3.set_xlim((0, 4))
    ax3.set_xlabel(r'$s_{\tau}$ [ms]')
    label_subplot(ax3, 'C', xoffset=-0.04, yoffset=-0.01)

    # -------- second row

    # Map positive peak of voltage
    ax4 = plt.subplot(234)
    ax4_h1 = ax4.scatter(x, y, c=max_V, s=10, marker='o', edgecolor='None', cmap='seismic')
    ax4.text(300, 300, 'positive peak', bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    voltage_color_bar(ax4_h1)
    cross_hair(ax4, x_AIS, y_AIS, color='white')
    set_axis_hidens(ax4)
    label_subplot(ax4, 'D', xoffset=-0.005, yoffset=-0.01)

    # Map of dendrite
    ax5 = plt.subplot(235)
    ax5.scatter(x, y, c=dendrite, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax5.text(300, 300, 'dendrite', bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    cross_hair(ax5, x_AIS, y_AIS, color='white')
    set_axis_hidens(ax5)
    label_subplot(ax5, 'E', xoffset=-0.005, yoffset=-0.01)

    # Std_delay vs max V for dendrite..
    ax6 = plt.subplot(236)
    ax6.scatter(std_delay_positive_peak, max_V, color='gray', marker='o', edgecolor='none', label = 'all')
    ax6.scatter(std_delay_positive_peak[np.where(axon)], max_V[np.where(axon)], color='blue', marker='^', edgecolor='none', label = 'axon')
    ax6.scatter(std_delay_positive_peak[np.where(dendrite)], max_V[np.where(dendrite)], color='red', marker='d', edgecolor='none', label = 'dendrite')
    ax6.legend(frameon=False)
    ax6.text(0.25, 45, 'positive peak', bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    ax6.set_ylim((0,50))
    ax6.set_ylabel(r'max $V$ [$\mu$V]')
    ax6.set_xlim((0,4))
    ax6.set_xlabel(r'$s_{\tau}$ [ms]')
    label_subplot(ax6, 'F', xoffset=-0.04, yoffset=-0.01)


    plt.show()


figure05()