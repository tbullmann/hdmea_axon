from hana.plotting import annotate_x_bar, set_axis_hidens
from hana.recording import half_peak_width, peak_peak_width, peak_peak_domain, DELAY_EPSILON, neighborhood, \
    electrode_neighborhoods, load_traces, load_positions
from hana.segmentation import __segment_axon, restrict_to_compartment
from publication.plotting import FIGURE_NEURON_FILE, without_spines_and_ticks, cross_hair, \
    legend_without_multiple_labels, label_subplot, plot_traces_and_delays, shrink_axes


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
    alpha=0.2
    N = np.random.normal(loc=1, scale=1.0, size=1000*(1-alpha))
    P = np.random.normal(loc=2, scale=1.0, size=1000*alpha)
    MixNP = np.hstack ((N, P))

    print (N0)
    print (MixNP)

    FPR, TPR = roc(N0, MixNP)

    print FPR


    ax1 = plt.subplot(221)

    ax1.plot(FPR,TPR,'b-', label='orig')
    ax1.plot((0,1),(alpha,1),'b--')

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

    # AP detection based on negative peak amplitude above threshold relative to noise level (Bakkum)
    mnr_N0 = pnr(Vbefore, type='min')
    mnr_NP = pnr(Vafter, type='min')

    mnr_FPR, mnr_TPR = roc(mnr_N0, mnr_NP)

    # AP detection based on neighborhood delays below threshold in valley (Bullmann)
    _, _, std_N0, _, _, _, _, _, _ = __segment_axon(t, Vbefore, neighbors)
    std_N0 = std_N0*2
    _, _, std_NP, _, _, _, _, _, _ = __segment_axon(t, V, neighbors)
    std_FPR, std_TPR = roc(std_N0, std_NP, type='smaller')

    alpha =0.3

    ax1 = plt.subplot(221)
    bins=np.linspace(0,10,num=100)
    plt.hist(mnr_N0, bins=bins, histtype='step', color='black', label='N')
    plt.hist(mnr_NP, bins=bins, histtype='step', color='red', label='N+P')
    plt.legend()
    ax1.set_ylabel ('count')
    ax1.set_xlabel ('negative peak/noise')

    ax2 = plt.subplot(222)
    bins = np.linspace(0, 4, num=100)
    plt.hist(std_N0, bins=bins, histtype='step', color='black', label='N')
    plt.hist(std_NP, bins=bins, histtype='step', color='red', label='N+P')
    plt.legend()
    ax2.set_ylabel ('count')
    ax2.set_xlabel ('std delay [ms]')

    ax3 = plt.subplot(223)
    ax3.plot(mnr_FPR,mnr_TPR,'b-', label='max/noise')
    ax3.plot(std_FPR,std_TPR,'g-', label='std delay')
    ax3.plot((0,1),(alpha,1),'k--', label='P/M=%1.3f' % alpha)
    ax3.set_xlabel ('FPR')
    ax3.set_ylabel ('"MPR"')
    ax3.legend(loc=4)
    ax3.set_aspect('equal')
    ax3.set_xlim((0,1))

    ax4 = plt.subplot(224)
    ax4.plot(mnr_FPR, mnr_TPR+(1-mnr_FPR)*(1-alpha),'b-', label='max/noise')
    ax4.plot(std_FPR, std_TPR+(1-std_FPR)*(1-alpha),'g-', label='std delay')
    ax4.set_xlabel ('FPR')
    ax4.set_ylabel ('TPR (estiamted)')
    ax4.legend(loc=4)
    ax4.set_aspect('equal')
    ax4.set_xlim((0,1))
    ax4.set_ylim((0,1))



    plt.show()

    # # Verbose axon segmentation function
    # delay, mean_delay, std_delay, expected_std_delay, thr, valid_delay, index_AIS, positive_delay, axon \
    #     = __segment_axon(t, V, neighbors)


def pnr(x, type='max'):
    """
    Using a robust estimator for standard deviation by using median absolute deviation (MAD):
    std (x) = 1.4826 * MAD (x)
    :param x:
    :return: Maximum to noise ratio (MNR)

    """
    robust_std = robust.mad(x, axis=1) * 1.4826
    if type=='max':
        peak = np.max(x, axis=1)
    if type=='min':
        peak = -np.min(x, axis=1)
    return peak / robust_std


test_rec()