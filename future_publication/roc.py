from hana.plotting import annotate_x_bar, set_axis_hidens
from hana.recording import half_peak_width, peak_peak_width, peak_peak_domain, DELAY_EPSILON, neighborhood, \
    electrode_neighborhoods, load_traces, load_positions
from hana.segmentation import segment_axon_verbose, restrict_to_compartment
from publication.plotting import FIGURE_NEURON_FILE, without_spines_and_ticks, cross_hair, \
    legend_without_multiple_labels, label_subplot, plot_traces_and_delays, shrink_axes

from scipy.stats import binom, beta, expon, norm
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from statsmodels import robust

import logging
logging.basicConfig(level=logging.DEBUG)


def figure05():
    # Load electrode coordinates
    neighbors = electrode_neighborhoods(mea='hidens')

    # Load example data
    # V, t, x, y, trigger, neuron = load_traces('data/neuron20.h5')
    V, t, x, y, trigger, neuron = load_traces('data/neuron11.h5')
    # V, t, x, y, trigger, neuron = load_traces('data/neuron5.h5')
    t *= 1000  # convert to ms

    Model1 = ModelDiscriminatorBakkum()
    Model1.fit(t, V)
    Model1.predict(pnr_threshold=5)
    Model2 = ModelDiscriminatorBullmann()
    Model2.fit(t, V, neighbors)
    Model2.predict()

    # Summary
    print ('Model I: ', Model1.summary())
    print ('Model II: ', Model2.summary())

    # Plotting
    fig = plt.figure('Figure Fits', figsize=(16, 10))
    fig.suptitle('Figure 5B. Fits for method I and II', fontsize=14, fontweight='bold')

    ax1 = plt.subplot(231)
    Model1.plot(ax1, xlabel=r'$\log_{10}(V_{n}/\sigma_{V})$')
    shrink_axes(ax1, xshrink=0.01)
    ax1.set_ylim((0,500))

    ax2 = plt.subplot(232)
    Model2.plot(ax2, xlabel=r'$s_{\tau}$ [ms]')
    shrink_axes(ax2, xshrink=0.01)
    ax2.set_ylim((0,500))

    ax3 = plt.subplot(233)
    Model1.plot_ROC(ax3, color='blue', marker='x', label = 'Method I')
    Model2.plot_ROC(ax3, color='black', marker='o', label = 'Method II')
    ax3.plot((0,1),(0,1), 'k--', label ='chance')
    ax3.legend(loc=4)

    ax4 = plt.subplot(234)
    Model1.plot_Map(ax4, x, y)
    ax4.text(300, 300, r'I: $V_{n} > %d\sigma_{V}; \tau > \tau_{AIS}$' % np.power(10, Model1.threshold),
            bbox=dict(facecolor='white', pad=5, edgecolor='none'), size=14)
    set_axis_hidens(ax4)

    ax5 = plt.subplot(235)
    Model2.plot_Map(ax5, x, y)
    ax5.text(300, 300, r'II: $s_{\tau} < s_{min}; \tau > \tau_{AIS}$', bbox=dict(facecolor='white', pad=5, edgecolor='none'), size=14)
    set_axis_hidens(ax5)

    plt.show()


# --- Functions


def AUC(FPR, TPR):
    """
    Calculate the AUC by using an average of a number of trapezoidal approximations for the given receiver operating
    characteristic (ROC) curve minus the area of chance level (diagonal, area = 0.5)
    Note: Taking the absolute value of np.trapz, because it result is negative iff FPR is in reverse order.
    :param FPR: False positive rate
    :param TPR: True positive rate
    :return: AUC = area under the ROC curve - 0.5 (area of chance)
    """
    area = abs(np.trapz(TPR, FPR))
    return area-0.5

def at_threshold(FPR, TPR, parameter, threshold):
    """
    False positive rate (FPR) and True positive rate (TPR) at the selected threshold.
    :param FPR: False positive rates of given receiver operating characteristic (ROC) curve
    :param TPR: True positive rate of given receiver operating characteristic (ROC) curve
    :param parameter: possible thresholds
    :param threshold: selected threshold
    """
    index = np.argmin(np.abs(parameter - threshold))
    FPR_at_threshold = FPR[index]
    TPR_at_threshold = TPR[index]
    return FPR_at_threshold, TPR_at_threshold


def smoothhist(x, bins=10, kernelsize=1):
    """
    Histogram smootheed with moving average with binomial distribution as kernel.
    Note: infinite values are excluded.
    :param x:
    :param bins:
    :param kernelsize: 1 (no smoothin)
    :return:
    """
    midpoints = np.convolve(bins / 2, np.ones(2), mode='valid')
    count, _ = np.histogram(x[np.where(np.isfinite(x))], bins)
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


def normcum(x):
    FPR = np.cumsum(x) / sum(x)
    return FPR


# ---- Classes

class ModelFunction (object):

    def __init__(self, formula_string = None, bounds_dict=None):
        self.variable_list = list(bounds_dict.keys())
        self.variable_string = ",".join(self.variable_list)
        self.bounds = zip(*bounds_dict.values())
        self.func = eval('lambda x, %s: %s' % (self.variable_string, formula_string))
        logging.info('Fitting y(x, %s) = %s' % (self.variable_string, formula_string))
        logging.info(bounds_dict)
        logging.info(self.variable_list)
        logging.info(self.bounds)

    def fit(self, x, y):
        params, cov = curve_fit(self.func, x, y, bounds=self.bounds)
        self.parameters = dict(zip(self.variable_list, params))
        logging.info(self.parameters)

    def predict(self, x, override=None):
        params = self.parameters.copy()
        if override:
            for key in override.keys():
                params[key] = override[key]
        return self.func(x, **params)


def test_fitting_a_model():
    model = model(formula_string='n * norm.pdf(x, loc, scale)',
                  bounds_dict=dict(n=[0, 11016], loc=[-.5, 2.], scale=[0, 10]))

    x = np.linspace(-1,1,15)
    y = model.func(x,n=1,loc=0,scale=0.2)

    model.fitvalues(x, y)

    xfit = np.linspace(-1,1,100)
    yfit = model.predict(xfit, override=dict(n=2))

    plt.plot(x,y,'x')
    plt.plot(xfit,yfit)
    plt.show()


class ModelDiscriminator(ModelFunction):

    def __init__(self, formula_string, bounds_dict, min_x=None, max_x=None, nbins=None):
        ModelFunction.__init__(self, formula_string, bounds_dict)
        self.min_x = min_x
        self.max_x = max_x
        self.nbins = nbins

    def fit(self, values):
        self.values = values
        bins = np.linspace(self.min_x, self.max_x, num=self.nbins)
        self.counts, self.midpoints = smoothhist(self.values, bins=bins)
        ModelFunction.fit(self, self.midpoints, self.counts)

    def predict(self, threshold, threshold_type='below'):
        self.threshold = threshold
        self.fitted_counts = ModelFunction.predict(self, self.midpoints)
        self.fitted_counts_N_only = ModelFunction.predict(self, self.midpoints, override=dict(n_P=0))
        self.fitted_counts_P_only = ModelFunction.predict(self, self.midpoints, override=dict(n_N=0))
        self.FPR = normcum(self.fitted_counts_N_only)
        self.TPR = normcum(self.fitted_counts_P_only)
        if threshold_type == 'above':
            self.FPR = 1 - self.FPR
            self.TPR = 1 - self.TPR
        self.FPR_at_threshold, self.TPR_at_threshold = at_threshold(self.FPR, self.TPR, self.midpoints, self.threshold)
        self.AUC = AUC(self.FPR, self.TPR)

    def plot(self, ax, xlabel = 'value', ylabel = 'counts'):
        ax.step(self.midpoints, self.counts, where='mid', color='gray', label='NP')
        ax.plot(self.midpoints, self.fitted_counts, color='gray', label='fit NP')
        ax.plot(self.midpoints, self.fitted_counts_N_only, color='red', label='fitted N')
        ax.plot(self.midpoints, self.fitted_counts_P_only, color='green', label='fitted P')
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def plot_ROC(self,ax, color='blue', marker='x', label = 'ROC'):
        ax.plot(self.FPR, self.TPR, color=color, label=label)
        ax.plot(self.FPR_at_threshold, self.TPR_at_threshold, color=color, marker=marker)
        ax.set_xlabel ('FPR')
        ax.set_ylabel ('TPR')

    def plot_Map(self, ax, x, y):
        if np.any(self.axon):
            ax.scatter(x, y, c=self.axon, s=10, marker='o', edgecolor='None', cmap='gray_r')

    def summary(self):
        n_N = sum(self.fitted_counts_N_only)
        n_P = sum(self.fitted_counts_P_only)
        gamma = n_P / (n_N + n_P)
        evaluation = dict(n_N=int(n_N), n_P=int(n_P), gamma=gamma,
                          FPR = self.FPR_at_threshold, TPR=self.TPR_at_threshold,
                          AUC=self.AUC)
        return evaluation


class ModelDiscriminatorBakkum(ModelDiscriminator):

    def __init__(self, nbins=200, min_x=-0.5, max_x=2., max_n=11016):
        ModelDiscriminator.__init__(self,
                               formula_string='n_N * norm.pdf(x, loc_N, scale_N) + n_P * norm.pdf(x, loc_P, scale_P)',
                               bounds_dict=dict(n_N=[0, max_n], loc_N=[min_x, max_x], scale_N=[0, 10],
                                                n_P=[0, max_n], loc_P=[min_x, max_x], scale_P=[0, 10]),
                               min_x=min_x, max_x=max_x, nbins=nbins)

    def fit(self, t, V):
        V_after_spike = V[:, t>0]
        values = np.log10(pnr(V_after_spike, type='min'))   # log10(amplitude negative peak / signal noise)
        ModelDiscriminator.fit(self, values)
        if self.parameters['loc_N'] > self.parameters['loc_P']:   # enforce loc_N < loc_P
            logging.info ('Swap Peaks')
            self.parameters['scale_N'], self.parameters['scale_P'] = self.parameters['scale_P'], self.parameters['scale_N']
            self.parameters['loc_N'], self.parameters['loc_P'] = self.parameters['loc_P'], self.parameters['loc_N']
            self.parameters['n_N'], self.parameters['n_P'] = self.parameters['n_P'], self.parameters['n_N']

    def predict(self, pnr_threshold=5):
        ModelDiscriminator.predict(self, threshold = np.log10(pnr_threshold), threshold_type='above')
        self.axon = self.values > self.threshold


class ModelDiscriminatorBullmann(ModelDiscriminator):
    def __init__(self, nbins=200, min_x=0, max_x=4, max_n=11016):
        ModelDiscriminator.__init__(self,
                                    formula_string='n_N * beta.pdf(x / %f, a_N, b_N) + n_P * expon.pdf(x / %f, 0, scale_P)' % (max_x, max_x),
                                    bounds_dict=dict(n_N=[0, max_n], a_N=[0, 10], b_N=[0, 10],
                                                     n_P=[0, max_n], scale_P=[0, 1]),
                                    min_x = min_x, max_x = max_x, nbins = nbins)

    def fit(self, t, V, neighbors):
        _, _, values, _, std_threshold, _, _, _, axon = segment_axon_verbose(t, V, neighbors)
        ModelDiscriminator.fit(self, values)
        self.threshold = std_threshold
        self.axon = axon

    def predict(self):
        ModelDiscriminator.predict(self, threshold=self.threshold, threshold_type='below')




figure05()