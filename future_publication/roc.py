from hana.plotting import annotate_x_bar, set_axis_hidens
from hana.recording import half_peak_width, peak_peak_width, peak_peak_domain, DELAY_EPSILON, neighborhood, \
    electrode_neighborhoods, load_traces, load_positions
from hana.segmentation import __segment_axon, restrict_to_compartment
from publication.plotting import FIGURE_NEURON_FILE, without_spines_and_ticks, cross_hair, \
    legend_without_multiple_labels, label_subplot, plot_traces_and_delays, shrink_axes

from scipy.stats import binom, beta, expon, norm
from scipy.optimize import curve_fit

import numpy as np
from matplotlib import pyplot as plt
from statsmodels import robust
import logging
logging.basicConfig(level=logging.DEBUG)


def fit_distribution():
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

    import numpy as np

    pnr_N0 = np.log10(pnr(Vbefore, type='min'))
    pnr_NP = np.log10(pnr(Vafter, type='min'))
    valid_peaks = pnr_NP > np.log10(pnr_threshold)
    pnr_thresholds, pnr_FPR, pnr_MPR = roc(pnr_N0, pnr_NP)

    # AP detection based on neighborhood delays below threshold in valley (Bullmann)
    _, _, std_N0, _, _, _, _, _, _ = __segment_axon(t, Vbefore, neighbors)
    std_N0 = std_N0 * 2
    _, _, std_NP, _, std_threshold, valid_delay, _, _, axon = __segment_axon(t, V, neighbors)
    std_thresholds, std_FPR, std_MPR = roc(std_N0, std_NP, type='smaller')



    # ------------- Method I
    N0 = pnr_N0
    NP = pnr_NP
    bins = np.linspace(-0.5, 2, num=nbins)

    counts = {}
    ydata, xdata = smoothhist(N0, bins=bins)

    def func_N0(x, loc, scale, n):
        return norm.pdf(x, loc, scale) * n

    popt, pcov = curve_fit(func_N0, xdata, ydata)
    # Constrain the optimization to the region of 0 < loc < 10, 0 < b < 10, 0 < n < 11016:
    popt, pcov = curve_fit(func_N0, xdata, ydata, bounds=([-0.5, 0., 0.], [2., 10., 11016.]))
    loc0, scale0, n0 = popt

    print loc0, scale0, n0

    # Plotting
    fig = plt.figure('Figure Fits', figsize=(16, 10))
    fig.suptitle('Figure 5B. Fits for method I and II', fontsize=14,
                 fontweight='bold')

    plt.subplot(221)
    plt.step(xdata, ydata, where='mid', color='gray', label='N0')
    plt.plot(xdata, func_N0(xdata, loc0, scale0, n0), color='gray', label='fit Norm(N0)')
    plt.legend()
    plt.xlabel (r'$\log_{10}(V_{n}/\sigma_{V})$')
    plt.ylabel ('count')

    ydata, xdata = smoothhist(NP, bins=bins, kernelsize=5)

    def func_NP(x, locN, scaleN, locP, scaleP, gamma, n):
        return ( norm.pdf(x, locN, scaleN) * (1-gamma) + norm.pdf(x, locP, scaleP) * gamma) * n

    popt, pcov = curve_fit(func_NP, xdata, ydata)
    # Constrain the optimization to the region of 0 < loc < 10, 0 < scale < 10, 0 < gamma < 0.5, 0 < n < 11016:
    popt, pcov = curve_fit(func_NP, xdata, ydata, bounds=([-0.5, 0., -0.5, 0., 0., 0.], [2., 10., 2., 10., 1, 11016.]))
    locN, scaleN, locP, scaleP, gamma, n = popt
    if locN>locP:  # if not positives loc > negatives loc then swap
        locP, locN = locN, locP
        scaleP, scaleN = scaleN, scaleP
        gamma = 1-gamma

    print locN, scaleN, locP, scaleP, gamma, n

    plt.subplot(222)
    plt.step(xdata, ydata, where='mid', color='gray', label='NP')
    plt.plot(xdata, func_NP(xdata, locN, scaleN, locP, scaleP, gamma, n), color='gray', label='fit Norm(N) + Norm(P)')
    plt.plot(xdata, norm.pdf(xdata, locN, scaleN) * (1-gamma) * n, color='red', label='fit Norm(N)')
    plt.plot(xdata, norm.pdf(xdata, locP, scaleP) * gamma * n, color='green', label='fit Norm(P)')
    plt.legend()
    plt.xlabel (r'$\log_{10}(V_{n}/\sigma_{V})$')
    plt.ylabel ('count')

    # ------------- Method II
    N0 = std_N0
    NP = std_NP
    bins = np.linspace(0, 4, num=nbins)

    counts = {}
    ydata, xdata = smoothhist(N0, bins=bins)
    xmax = 4    # to rescale x to 0..1


    def func_N0(x, a, b, n):
        return beta.pdf(x / xmax, a, b) * n

    popt, pcov = curve_fit(func_N0, xdata, ydata)
    # Constrain the optimization to the region of 0 < a < 10, 0 < b < 10, 0 < n < 11016:
    popt, pcov = curve_fit(func_N0, xdata, ydata, bounds=(0, [10.,10.,11016.]))
    a0,b0,n0 = popt

    print a0,b0,n0


    plt.subplot(223)
    plt.step(xdata, ydata, where='mid', color='gray', label='N0')
    plt.plot(xdata, func_N0(xdata, a0, b0, n0), color='gray', label='fit Beta(N0)')
    plt.legend()
    plt.xlabel (r'$s_{\tau}$ [ms]')
    plt.ylabel ('count')


    ydata, xdata = smoothhist(NP, bins=bins, kernelsize=5)

    def func_NP3(x, an, bn, scale, gamma, n):
        return (beta.pdf(x / xmax, an, bn) * (1 - gamma) + expon.pdf(x / xmax,0,scale) * gamma) * n

    popt, pcov = curve_fit(func_NP3, xdata, ydata)
    # Constrain the optimization to the region of 0 < a < 10, 0 < b < 10, 0<-loc<10, 0<scale<1, 0 < gamma < 1, 0 < n < 11016:
    popt, pcov = curve_fit(func_NP3, xdata, ydata, bounds=(0, [10., 10., 1., 1., 11016.]))
    an, bn, scale, gamma, n = popt

    print an, bn, scale, gamma, n

    plt.subplot(224)
    plt.step(xdata, ydata, where='mid', color='gray', label='NP')
    plt.plot(xdata, func_NP3(xdata, an, bn, scale, gamma, n), color='gray', label='fit Beta(N) + Expon(P)')
    plt.plot(xdata, beta.pdf(xdata/xmax, an, bn) * (1 - gamma) * n, color='red', label='fit Beta(N)')
    plt.plot(xdata, expon.pdf(xdata/xmax, 0, scale) * gamma * n, color='green', label='fit Expon(P)')
    plt.legend()
    plt.xlabel (r'$s_{\tau}$ [ms]')
    plt.ylabel ('count')
    plt.show()

    #TODO Calculate gamma from sum(P)/(sum(N)+sum(P)); it is not integral expon.pdf = 1 (as for the others)


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
    _, _, std_N0, _, _, _, _, _, _ = __segment_axon(t, Vbefore, neighbors)
    std_N0 = std_N0*2
    _, _, std_NP, _, std_threshold, valid_delay, _, _, axon = __segment_axon(t, V, neighbors)
    std_thresholds, std_FPR, std_MPR = roc(std_N0, std_NP, type='smaller')

    # Plotting
    fig = plt.figure('Figure ROC', figsize=(16, 10))
    fig.suptitle('Figure 5. Comparison of segmentation methods.', fontsize=14,
                 fontweight='bold')

    ax1 = plt.subplot(231)
    bins = np.linspace(-0.5, 2, num=nbins)
    midpoints, pnr_counts, pnr_gamma = unmix_NP(pnr_N0, pnr_NP, bins)
    plot_distributions(ax1, midpoints, pnr_counts)
    ax1.annotate('threshold \n$%d\sigma_{V}$' % pnr_threshold, xy=(np.log10(pnr_threshold), 0), xytext=(np.log10(pnr_threshold), 200),
             arrowprops=dict(facecolor='black', width=1), size=14)
    ax1.set_ylim((0,500))
    ax1.set_xlabel (r'$\log_{10}(V_{n}/\sigma_{V})$')
    ax1.text(-0.3,450, 'I', size=14)

    ax2 = plt.subplot(232)
    bins = np.linspace(0, 4, num=nbins)
    midpoints, std_counts, std_gamma = unmix_NP(std_N0, std_NP, bins)
    plot_distributions(ax2, midpoints, std_counts)
    ax2.annotate('threshold \n$s_{min}=%1.3f$ms' % std_threshold, xy=(std_threshold, 0), xytext=(std_threshold, 200),
                 arrowprops=dict(facecolor='black', width=1), size=14)
    ax2.set_ylim((0,500))
    ax2.set_xlabel (r'$s_{\tau}$ [ms]')
    ax2.text(0.3,450, 'II', size=14)

    ax3 = plt.subplot(233)
    plot_roc(ax3, gamma, pnr_FPR, pnr_MPR, pnr_threshold, pnr_thresholds, color='b', marker='x', label='I')
    plot_roc(ax3, gamma, std_FPR, std_MPR, std_threshold, std_thresholds, color='k', marker='o', label='II')
    ax3.set_xlabel('FPR')
    ax3.set_ylabel('TPR')
    ax3.legend(loc=4, scatterpoints=1)
    ax3.set_aspect('equal')
    ax3.set_xlim((0, 1))
    ax3.set_ylim((0, 1))

    ax4 = plt.subplot(234)
    ax4.scatter(x, y, c=valid_peaks, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax4.text(300, 300, r'I: $V_{n} > %d\sigma_{V}; \tau > \tau_{AIS}$' % pnr_threshold , bbox=dict(facecolor='white', pad=5, edgecolor='none'), size=14)
    set_axis_hidens(ax4)

    ax5 = plt.subplot(235)
    ax5.scatter(x, y, c=axon, s=10, marker='o', edgecolor='None', cmap='gray_r')
    ax5.text(300, 300, r'II: $s_{\tau} < s_{min}; \tau > \tau_{AIS}$', bbox=dict(facecolor='white', pad=5, edgecolor='none'), size=14)
    set_axis_hidens(ax5)

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




fit_distribution()

# figure05()