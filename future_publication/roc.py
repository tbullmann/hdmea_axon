from hana.plotting import annotate_x_bar, set_axis_hidens
from hana.recording import half_peak_width, peak_peak_width, peak_peak_domain, DELAY_EPSILON, neighborhood, \
    electrode_neighborhoods, load_traces, load_positions
from hana.segmentation import __segment_axon, restrict_to_compartment
from publication.plotting import FIGURE_NEURON_FILE, without_spines_and_ticks, cross_hair, \
    legend_without_multiple_labels, label_subplot, plot_traces_and_delays, shrink_axes

from scipy.stats import binom
import numpy as np
from matplotlib import pyplot as plt
from statsmodels import robust
import logging
logging.basicConfig(level=logging.DEBUG)








def roc (N,P, type='greater'):
    # assuming P>N
    xy = (np.hstack((N, P)))
    labelsxy = (np.hstack((np.ones_like(N), np.zeros_like(P))))
    if type=='greater': # argument of sort in descending order of values thus P > threshold > N
        index = np.argsort(xy)[::-1]
    if type == 'smaller':  # argument of sort in ascending order of values thus P < threshold < N
        index = np.argsort(xy)
    labelsxy = labelsxy[index]
    FPR = np.cumsum(labelsxy)/len(N)
    TPR = np.cumsum(np.ones_like(labelsxy)-labelsxy)/len(P)
    return FPR, TPR




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




def test_rec():
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

    pnr_FPR, pnr_TPR = roc(pnr_N0, pnr_NP)

    # AP detection based on neighborhood delays below threshold in valley (Bullmann)
    _, _, std_N0, _, _, _, _, _, _ = __segment_axon(t, Vbefore, neighbors)
    std_N0 = std_N0*2
    _, _, std_NP, _, std_threshold, valid_delay, _, _, axon = __segment_axon(t, V, neighbors)
    std_FPR, std_TPR = roc(std_N0, std_NP, type='smaller')

    fig = plt.figure('Figure ROC', figsize=(16, 10))
    fig.suptitle('Figure ROC. Comparison of segmentation methods', fontsize=14,
                 fontweight='bold')

    ax1 = plt.subplot(231)
    bins = np.linspace(-0.5, 2, num=nbins)
    midpoints, pnr_counts, pnr_gamma = unmix_NP(pnr_N0, pnr_NP, bins)
    plot_distributions(ax1, midpoints, pnr_counts)
    ax1.set_ylim((0,500))
    ax1.set_xlabel (r'$\log_{10}(V_{n}/\sigma_{V})$')
    ax1.text(-0.3,450, 'I', size=14)

    ax2 = plt.subplot(232)
    bins = np.linspace(0, 4, num=nbins)
    midpoints, std_counts, std_gamma = unmix_NP(std_N0, std_NP, bins)
    plot_distributions(ax2, midpoints, std_counts)
    ax2.set_ylim((0,500))
    ax2.set_xlabel (r'$s_{\tau}$ [ms]')
    ax2.text(0.3,450, 'II', size=14)

    ax3 = plt.subplot(233)
    ax3.plot((0,1),(gamma,1),'k--', label=r'$\gamma$=%1.3f' % gamma)
    ax3.plot(pnr_FPR,pnr_TPR,'b-', label='I')
    ax3.plot((0,1),(pnr_gamma,1),'b--', label=r'$\gamma$=%1.3f' % pnr_gamma)
    ax3.plot(std_FPR,std_TPR,'g-', label='II')
    ax3.plot((0,1),(std_gamma,1),'g--', label=r'$\gamma$=%1.3f' % std_gamma)
    ax3.set_xlabel ('FPR')
    ax3.set_ylabel ('"MPR"')
    ax3.legend(loc=4)
    ax3.set_aspect('equal')
    ax3.set_xlim((0,1))

    ax4 = plt.subplot(234)
    ax4.scatter(x, y, c=valid_peaks, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax4.text(300, 300, r'I: $V_{n} > %d\sigma_{V}; \tau > \tau_{AIS}$' % pnr_threshold, bbox=dict(facecolor='white', pad=5, edgecolor='none'), size=14)
    set_axis_hidens(ax4)

    ax5 = plt.subplot(235)
    ax5.scatter(x, y, c=axon, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax5.text(300, 300, r'II: $s_{\tau} < s_{min}; \tau > \tau_{AIS}$', bbox=dict(facecolor='white', pad=5, edgecolor='none'), size=14)
    set_axis_hidens(ax5)

    ax6 = plt.subplot(236)
    ax6.plot(pnr_FPR, pnr_TPR+(1-pnr_FPR)*(1-pnr_gamma),'b:', label=r'I, $\gamma$=%1.3f' % pnr_gamma)
    ax6.plot(pnr_FPR, pnr_TPR+(1-pnr_FPR)*(1-gamma),'b-', label=r'I, $\gamma$=%1.3f' % gamma)
    ax6.plot(std_FPR, std_TPR+(1-std_FPR)*(1-std_gamma),'g:', label=r'II, $\gamma$=%1.3f' % std_gamma)
    ax6.plot(std_FPR, std_TPR+(1-std_FPR)*(1-gamma),'g-', label=r'II, $\gamma$=%1.3f' % gamma)
    ax6.set_xlabel ('FPR')
    ax6.set_ylabel ('TPR ($\gamma$)')
    ax6.legend(loc=4)
    ax6.set_aspect('equal')
    ax6.set_xlim((0,1))
    ax6.set_ylim((0,1))



    plt.show()

    # # Verbose axon segmentation function
    # delay, mean_delay, std_delay, expected_std_delay, thr, valid_delay, index_AIS, positive_delay, axon \
    #     = __segment_axon(t, V, neighbors)


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


def unmix_NP(N0, NP, bins):
    counts = {}
    counts['N0'], midpoints = smoothhist(N0, bins=bins)
    counts['NP'], midpoints = smoothhist(NP, bins=bins)

    beta = counts['NP'][np.argmax(counts['N0'])] / np.max(counts['N0'])  # fit peaks
    gamma = 1-beta  # proportion of gamma = P/(N+P)

    counts['N'] = counts['N0'] * beta
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


test_rec()