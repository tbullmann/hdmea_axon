from hana.segmentation import segment_dendrite_verbose, segment_axon_verbose
from publication.plotting import FIGURE_NEURON_FILE, cross_hair, label_subplot, voltage_color_bar, adjust_position
from hana.plotting import mea_axes
from hana.recording import electrode_neighborhoods, load_traces

from scipy.stats import binom
import numpy as np
from matplotlib import pyplot as plt
from statsmodels import robust
import logging
logging.basicConfig(level=logging.DEBUG)



# Final figure 5

def figure05():
    # Load electrode coordinates
    neighbors = electrode_neighborhoods(mea='hidens')

    # Load example data
    V, t, x, y, trigger, neuron = load_traces(FIGURE_NEURON_FILE)
    t *= 1000  # convert to ms

    Vbefore = V[:, :80]
    Vafter = V[:, 81:]

    gamma = 0.3
    pnr_threshold = 5
    nbins = 200

    # AP detection based on negative peak amplitude above threshold relative to noise level (Bakkum)
    pnr_N0 = np.log10(pnr(Vbefore, type='min'))
    pnr_NP = np.log10(pnr(Vafter, type='min'))
    valid_peaks = pnr_NP > np.log10(pnr_threshold)
    pnr_thresholds, pnr_FPR, pnr_MPR = roc(pnr_N0, pnr_NP)

    # AP detection based on neighborhood delays below threshold in valley (Bullmann)
    _, _, std_N0, _, _, _, _, _, _ = segment_axon_verbose(t, Vbefore, neighbors)
    std_N0 = std_N0*2
    _, _, std_NP, _, std_threshold, valid_delay, _, _, axon = segment_axon_verbose(t, V, neighbors)
    std_thresholds, std_FPR, std_MPR = roc(std_N0, std_NP, type='smaller')

    # Plotting
    fig = plt.figure('Figure 5', figsize=(16, 10))
    fig.suptitle('Figure 5. Comparison of segmentation methods.', fontsize=14,
                 fontweight='bold')

    ax1 = plt.subplot(231)
    bins = np.linspace(-0.5, 2, num=nbins)
    midpoints, pnr_counts, pnr_gamma = unmix_NP(pnr_N0, pnr_NP, bins)
    plot_distributions(ax1, midpoints, pnr_counts)
    ax1.annotate('(fixed) threshold \n$%d\sigma_{V}$' % pnr_threshold, xy=(np.log10(pnr_threshold), 0), xytext=(np.log10(pnr_threshold), 200),
             arrowprops=dict(facecolor='black', width=1), size=14)
    ax1.set_ylim((0,500))
    ax1.set_xlabel (r'$\log_{10}(V_{n}/\sigma_{V})$')
    ax1.text(-0.3,450, 'I', size=14)
    adjust_position(ax1, xshrink=0.01)
    label_subplot(ax1, 'A', xoffset=-0.05, yoffset=-0.01)

    ax2 = plt.subplot(232)
    bins = np.linspace(0, 4, num=nbins)
    midpoints, std_counts, std_gamma = unmix_NP(std_N0, std_NP, bins)
    plot_distributions(ax2, midpoints, std_counts)
    ax2.annotate('(adaptive) threshold \n$s_{min}=%1.3f$ms' % std_threshold, xy=(std_threshold, 0), xytext=(std_threshold, 200),
                 arrowprops=dict(facecolor='black', width=1), size=14)
    ax2.set_ylim((0,500))
    ax2.set_xlabel (r'$s_{\tau}$ [ms]')
    ax2.text(0.3,450, 'II', size=14)
    adjust_position(ax2, xshrink=0.01)
    label_subplot(ax2, 'B', xoffset=-0.05, yoffset=-0.01)

    ax3 = plt.subplot(233)
    plot_roc(ax3, gamma, pnr_FPR, pnr_MPR, pnr_threshold, pnr_thresholds, color='b', marker='x', label='I')
    plot_roc(ax3, gamma, std_FPR, std_MPR, std_threshold, std_thresholds, color='k', marker='o', label='II')
    ax3.set_xlabel('FPR')
    ax3.set_ylabel('TPR')
    ax3.legend(loc=4, scatterpoints=1)
    ax3.set_aspect('equal')
    ax3.set_xlim((0, 1))
    ax3.set_ylim((0, 1))
    label_subplot(ax3, 'C', xoffset=-0.04, yoffset=-0.01)

    ax4 = plt.subplot(234)
    ax4.scatter(x, y, c=valid_peaks, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax4.text(300, 300, r'I: $V_{n} > %d\sigma_{V}; \tau > \tau_{AIS}$' % pnr_threshold , bbox=dict(facecolor='white', pad=5, edgecolor='none'), size=14)
    mea_axes(ax4)
    label_subplot(ax4, 'D', xoffset=-0.03, yoffset=-0.01)

    ax5 = plt.subplot(235)
    ax5.scatter(x, y, c=axon, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax5.text(300, 300, r'II: $s_{\tau} < s_{min}; \tau > \tau_{AIS}$', bbox=dict(facecolor='white', pad=5, edgecolor='none'), size=14)
    mea_axes(ax5)
    label_subplot(ax5, 'E', xoffset=-0.03, yoffset=-0.01)

    plt.show()


def roc (N,P, type='greater'):
    # assuming P>N
    xy = (np.hstack((N, P)))
    labelsxy = (np.hstack((np.ones_like(N), np.zeros_like(P))))
    if type=='greater': # argument of sort in descending order of values thus P > threshold > N
        index = np.argsort(xy)[::-1]
    if type == 'smaller':  # argument of sort in ascending order of values thus P < threshold < N
        index = np.argsort(xy)
    threshold = xy[index]
    labelsxy = labelsxy[index]
    FPR = np.cumsum(labelsxy)/len(N)
    TPR = np.cumsum(np.ones_like(labelsxy)-labelsxy)/len(P)
    return threshold, FPR, TPR


def test_roc():
    N0 = np.random.normal(loc=1, scale=1.0, size=1000)
    gamma=0.2
    N = np.random.normal(loc=1, scale=1.0, size=1000*(1-gamma))
    P = np.random.normal(loc=2, scale=1.0, size=1000*gamma)
    MixNP = np.hstack ((N, P))

    print (N0)
    print (MixNP)

    FPR, TPR = roc(N0, MixNP)

    print FPR

    ax1 = plt.subplot(221)

    ax1.plot(FPR,TPR,'b-', label='orig')
    ax1.plot((0,1),(gamma,1),'b--')

    ax1.legend(loc=4)
    ax1.set_aspect('equal')
    ax1.set_xlim((0,1))

    plt.subplot(222)
    bins=np.linspace(-4,8,num=50)
    plt.hist(N0, bins=bins, histtype='step', color='black', label='N0')
    plt.hist(MixNP, bins=bins, histtype='step', color='red', label='Mix')

    plt.show()


def plot_roc(ax3, gamma, pnr_FPR, pnr_MPR, pnr_threshold, pnr_thresholds, color = 'b', marker = 'x', label='I'):
    pnr_TPR = estimate_TPR(gamma, pnr_FPR, pnr_MPR)
    pnr_FPR_at_threshold, pnr_TPR_at_threshold = at_threshold(pnr_FPR, pnr_TPR, pnr_thresholds, pnr_threshold)
    print ('FPR=%1.6f, TPR=%1.6f' % (pnr_FPR_at_threshold, pnr_TPR_at_threshold))
    ax3.plot(pnr_FPR, pnr_TPR, color, label=label + r', $\gamma$=%1.3f' % gamma)
    ax3.scatter(pnr_FPR_at_threshold, pnr_TPR_at_threshold, color=color, marker=marker, label=label+' threshold')


def at_threshold(FPR, TPR, parameter, threshold):
    index = np.argmin(np.abs(parameter - threshold))
    FPR_at_threshold = FPR[index]
    TPR_at_threshold = TPR[index]
    return FPR_at_threshold, TPR_at_threshold


def estimate_TPR(gamma, FPR, MPR):
    """If the proportion gamma = P/(N+P) = P/M in the mix M=T+P is known, TPR can be estimated from the MPR."""
    estiamted_TPR = MPR + (1 - FPR) * (1 - gamma)
    return estiamted_TPR


def plot_distributions(ax1, midpoints, counts):
    # # using lines between midpoints
    # ax1.plot(midpoints, counts['N0'], color='gray', label='N0')
    # ax1.plot(midpoints, counts['NP'], color='black', label='N+P')
    # ax1.plot(midpoints, counts['N'], color='red', label='N')
    # ax1.plot(midpoints, counts['P'], color='green', label='P')
    # using steps
    ax1.step(midpoints, counts['N0'], where='mid', color='gray', label='N0')
    ax1.step(midpoints, counts['NP'], where='mid', color='black', label='N+P')
    ax1.step(midpoints, counts['N'], where='mid', color='red', label='N')
    ax1.step(midpoints, counts['P'], where='mid', color='green', label='P')
    ax1.legend()
    ax1.set_ylabel('count')


def unmix_NP(N0, NP, bins, gamma=None ):
    counts = {}
    counts['N0'], midpoints = smoothhist(N0, bins=bins)
    counts['NP'], midpoints = smoothhist(NP, bins=bins)

    if not gamma:      # proportion of positives, gamma = P/(N+P)
        gamma = 1 - counts['NP'][np.argmax(counts['N0'])] / np.max(counts['N0'])  # fit peaks
    epsilon = 1-gamma  # proportion of negative, epsilon = N/(N+P)

    counts['N'] = counts['N0'] * epsilon
    counts['P'] = counts['NP'] - counts['N']
    return midpoints, counts, gamma


def smoothhist(x, bins=10, kernelsize=1):
    """histogram smootheed with moving average with binomial distribution as kernel"""
    midpoints = np.convolve(bins / 2, np.ones(2), mode='valid')
    count, _ = np.histogram(x, bins)
    kernel = binom.pmf(np.linspace(0, kernelsize - 1, kernelsize), kernelsize - 1, 0.5)
    count = np.convolve(count, kernel, mode='same')
    return count, midpoints


def pnr(x, type='max'):
    """
    Using a robust estimator for standard deviation by using median absolute deviation (MAD):
    std (x) = 1.4826 * MAD (x)
    :param x:
    :return: Maximum to noise ratio (MNR)

    """
    robust_std = robust.mad(x, axis=1) * 1.4826
    if type=='max':
        peak = np.max(x, axis=1)-np.median(x, axis=1)
    if type=='min':
        peak = -(np.min(x, axis=1)-np.median(x, axis=1))
    return peak / robust_std


# Old figure 5

def figure05_old():

    # Load electrode coordinates
    neighbors = electrode_neighborhoods(mea='hidens')

    # Load example data
    V, t, x, y, trigger, neuron = load_traces(FIGURE_NEURON_FILE)
    t *= 1000  # convert to ms

    # Minimum Voltage and verbose axon segmentation function and
    min_V = np.min(V, axis=1)
    _, _, std_delay_negative_peak, _, thr_std_delay_negative_peak, _, index_AIS, _, axon \
        = segment_axon_verbose(t, V, neighbors)

    # Maximum Voltage and verbose dendrite segmentation function and
    max_V = np.max(V, axis=1)
    _, _, std_delay_positive_peak, _, thr_std_delay_positive_peak, _, _, _, _, _, dendrite \
        =segment_dendrite_verbose(t, V, neighbors)

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
    voltage_color_bar(ax1_h1, label=r'$V_n$ [$\mu$V]')
    cross_hair(ax1, x_AIS, y_AIS, color='white')
    mea_axes(ax1)
    label_subplot(ax1, 'A', xoffset=-0.005, yoffset=-0.01)

    # Map of axon
    ax2 = plt.subplot(232)
    ax2.scatter(x, y, c=axon, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax2.text(300, 300, 'axon', bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    cross_hair(ax2, x_AIS, y_AIS, color='white')
    mea_axes(ax2)
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
    voltage_color_bar(ax4_h1, label=r'$V_p$ [$\mu$V]')
    cross_hair(ax4, x_AIS, y_AIS, color='white')
    mea_axes(ax4)
    label_subplot(ax4, 'D', xoffset=-0.005, yoffset=-0.01)

    # Map of dendrite
    ax5 = plt.subplot(235)
    ax5.scatter(x, y, c=dendrite, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax5.text(300, 300, 'dendrite', bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    cross_hair(ax5, x_AIS, y_AIS, color='white')
    mea_axes(ax5)
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

# figure05_old()
figure05()