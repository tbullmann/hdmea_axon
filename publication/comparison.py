import logging

import numpy as np
from scipy.stats import binom
from statsmodels import robust

from hana.misc import ModelFunction
from hana.segmentation import segment_axon_verbose


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
    """
    Normalized cummulative sum, thus values adding up to 1.
    """
    FPR = np.cumsum(x) / sum(x)
    return FPR


class ModelDiscriminator(ModelFunction):
    """
    Base class for an discriminator for data modeled as a mixture of two distributions.
    """

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
        self.fitted_counts_N_only = ModelFunction.predict(self, self.midpoints, override=dict(amp_P=0))
        self.fitted_counts_P_only = ModelFunction.predict(self, self.midpoints, override=dict(amp_N=0))
        self.FPR = normcum(self.fitted_counts_N_only)
        self.TPR = normcum(self.fitted_counts_P_only)
        if threshold_type == 'above':
            self.FPR = 1 - self.FPR
            self.TPR = 1 - self.TPR
        self.FPR_at_threshold, self.TPR_at_threshold = at_threshold(self.FPR, self.TPR, self.midpoints, self.threshold)
        self.AUC = AUC(self.FPR, self.TPR)

    def plot(self, ax, xlabel = 'value', ylabel = 'counts'):
        ax.step(self.midpoints, self.counts, where='mid', color='gray', label='NP')
        ax.plot(self.midpoints, self.fitted_counts, color='black', label='fit NP')
        ax.plot(self.midpoints, self.fitted_counts_N_only, color='red', label='fitted N')
        ax.plot(self.midpoints, self.fitted_counts_P_only, color='green', label='fitted P')
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def plot_ROC(self,ax, color='blue', marker='x', label = 'ROC'):
        ax.plot(self.FPR, self.TPR, color=color, label=label)
        ax.scatter(self.FPR_at_threshold, self.TPR_at_threshold, color=color, marker=marker, label='%s threshold' % label)
        ax.set_xlabel ('FPR')
        ax.set_ylabel ('TPR')

    def plot_Map(self, ax, x, y):
        if np.any(self.axon):
            ax.scatter(x, y, c=self.axon, s=10, marker='o', edgecolor='None', cmap='gray_r')

    def summary(self, subject, method):
        n_N = sum(self.fitted_counts_N_only)
        n_P = sum(self.fitted_counts_P_only)
        gamma = n_P / (n_N + n_P)
        evaluation = dict(subject=subject, method=method,
                          n_N=int(n_N), n_P=int(n_P), gamma=gamma,
                          FPR = self.FPR_at_threshold, TPR=self.TPR_at_threshold,
                          AUC=self.AUC)
        return evaluation


class ModelDiscriminatorBakkum(ModelDiscriminator):
    """
    Discriminator as described in Bakkum et al, 2013, Nat Commun
    """

    def __init__(self, nbins=200, min_x=-0.5, max_x=2., max_n=11016, pnr_threshold=5):
        ModelDiscriminator.__init__(self,
            formula_string='amp_N * norm.pdf(x, loc_N, scale_N) + amp_P * norm.pdf(x, loc_P, scale_P)',
            bounds_dict=dict(amp_N=[0, max_n], loc_N=[0, max_x], scale_N=[0, 10],
                            amp_P=[0, max_n], loc_P=[0.4, max_x], scale_P=[0, 10]),
            min_x=min_x, max_x=max_x, nbins=nbins)

    def fit(self, t, V):
        V_after_spike = V[:, t>0]
        values = np.log10(pnr(V_after_spike, type='min'))   # log10(amplitude negative peak / signal noise)
        ModelDiscriminator.fit(self, values)
        if self.parameters['loc_N'] > self.parameters['loc_P']:   # enforce loc_N < loc_P
            logging.info ('Swap Peaks')
            self.parameters['scale_N'], self.parameters['scale_P'] = self.parameters['scale_P'], self.parameters['scale_N']
            self.parameters['loc_N'], self.parameters['loc_P'] = self.parameters['loc_P'], self.parameters['loc_N']
            self.parameters['amp_N'], self.parameters['amp_P'] = self.parameters['amp_P'], self.parameters['amp_N']

    def predict(self, pnr_threshold=5):
        ModelDiscriminator.predict(self, threshold = np.log10(pnr_threshold), threshold_type='above')
        self.axon = self.values > self.threshold


class ModelDiscriminatorBullmann(ModelDiscriminator):
    """
    Discriminator as described in Bullmann et al, submitted
    """
    def __init__(self, nbins=200, min_x=0, max_x=4, max_n=11016):
        ModelDiscriminator.__init__(self,
            formula_string='amp_N * beta.pdf(x / %f, a_N, b_N) + amp_P * expon.pdf(x / %f, 0, scale_P)' % (max_x, max_x),
            bounds_dict=dict(amp_N=[0, max_n], a_N=[0, 10], b_N=[0, 10],
                             amp_P=[0, max_n], scale_P=[0, 0.2]),
            min_x = min_x, max_x = max_x, nbins = nbins)

    def fit(self, t, V, neighbors):
        _, _, values, _, std_threshold, _, _, _, axon = segment_axon_verbose(t, V, neighbors)
        ModelDiscriminator.fit(self, values)
        self.threshold = std_threshold
        self.axon = axon

    def predict(self):
        ModelDiscriminator.predict(self, threshold=self.threshold, threshold_type='below')