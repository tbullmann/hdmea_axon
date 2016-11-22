from hana.matlab import load_traces, load_positions
from hana.plotting import annotate_x_bar, set_axis_hidens
from hana.recording import half_peak_width, peak_peak_width, peak_peak_domain
from publication.plotting import FIGURE_NEURON_FILE, FIGURE_ELECTRODES_FILE, without_spines_and_ticks, cross_hair, \
    legend_without_multiple_labels, label_subplot

import numpy as np
from matplotlib import pyplot as plt

import logging
logging.basicConfig(level=logging.DEBUG)


# Testing code

def testing_load_traces():
    V, t, x, y, trigger, neuron = load_traces(FIGURE_NEURON_FILE)
    Vtrigger = V[int(trigger)]

    hpw = half_peak_width(t, Vtrigger)
    print hpw

    ppw = peak_peak_width(t, Vtrigger)
    print ppw


# Final figure 2

def figure02(testing=False):
    fig = plt.figure('Figure 2', figsize=(12,9))

    pos = load_positions(FIGURE_ELECTRODES_FILE)  # only used for set_axis_hidens

    V, t, x, y, trigger, neuron = load_traces(FIGURE_NEURON_FILE)
    t *= 1000  # convert to ms

    # Electrode with most minimal V corresponding to proximal AIS, get coordinates and recorded voltage trace
    ais = np.unravel_index(np.argmin(V), V.shape)[0]
    xais = x[ais]
    yais = y[ais]
    Vais = V[int(ais)]

    if testing:  # matplotlib is slow, plotting all traces takes 30 sec
        V = V[range(ais-10,ais)+range(ais+1,ais+11)]       # 20 traces only
        ais = np.unravel_index(np.argmin(V), V.shape)[0]   # get "new" AIS index because old one is shifted

    # align negative peak
    indicies_neg_peak = np.argmin(V,axis=1)
    t_neg_peak = t[indicies_neg_peak]
    index_ais_neg_peak = indicies_neg_peak[ais]
    Vshifted = np.array([np.roll(row, shift) for row, shift in zip(V, index_ais_neg_peak - indicies_neg_peak - 1 )])
    Vshiftedais = Vshifted[ais]

    # subplot original unaligned traces
    ax1 = plt.subplot(221)
    ax1.plot(t, V.T,'-', color='gray', label='unaligned')
    ax1.plot(t, Vais, 'k-', label='(proximal) AIS')
    annotate_x_bar(peak_peak_domain(t, Vais), min(Vais)/2, text=' $\delta_p$ = %0.3f ms' % peak_peak_width(t, Vais))
    legend_without_multiple_labels(ax1, loc=4, frameon=False)
    ax1.set_xlim((-0.5,4))
    ax1.set_ylabel(r'V [$\mu$V]')
    ax1.set_xlabel(r'$\Delta$t [ms]')
    without_spines_and_ticks(ax1)
    label_subplot(ax1, 'A')

    # subplot delay map
    ax2 = plt.subplot(222)
    h1 = ax2.scatter(x, y, c=t_neg_peak, s=10, marker='o', edgecolor='None', cmap='gray')
    h2 = plt.colorbar(h1)
    h2.set_label(r'$\tau$ [ms]')
    cross_hair(ax2, xais, yais)
    set_axis_hidens(ax2, pos)
    label_subplot(ax2, 'B', xoffset=-0.02)


    # subplot original unaligned traces
    ax3 = plt.subplot(223)
    ax3.plot(t, Vshifted.T,'-', color='gray', label='aligned')
    ax3.plot(t, Vshiftedais, 'k-', label='(proximal) AIS')
    legend_without_multiple_labels(ax3, loc=4, frameon=False)
    ax3.set_xlim((-0.5,4))
    ax3.set_ylabel(r'V [$\mu$V]')
    ax3.set_xlabel(r'$\Delta$t [ms]')
    without_spines_and_ticks(ax3)
    label_subplot(ax3, 'C')


    # subplot histogram of delays
    ax4 = plt.subplot(224)
    ax4.hist(t_neg_peak, bins=len(t), facecolor='gray', edgecolor='gray')
    ax4.vlines(0, 0, 180, color='k', linestyles=':')
    ax4.hlines(len(x)/len(t), min(t), max(t), color='k', linestyles='--')
    ax4.set_ylabel(r'count')
    ax4.set_xlabel(r'$\Delta$t [ms]')
    without_spines_and_ticks(ax4)
    label_subplot(ax4, 'D')

    plt.show()


# testing_load_traces()
figure02()
