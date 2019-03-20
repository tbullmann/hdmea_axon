import logging
import os

import numpy as np
from matplotlib import pyplot as plt

from hana.plotting import mea_axes
from hana.recording import electrode_neighborhoods
from publication.comparison import ModelDiscriminatorBakkum, ModelDiscriminatorBullmann
from publication.data import FIGURE_CULTURE, FIGURE_NEURON
from publication.experiment import Experiment
from publication.plotting import show_or_savefig, adjust_position, plot_pairwise_comparison, without_spines_and_ticks
from data import FIGURE_NEURONS

logging.basicConfig(level=logging.DEBUG)


def make_figure(figurename, figpath=None):

    V, t, x, y, trigger, neuron = Experiment(FIGURE_CULTURE).traces(FIGURE_NEURON)
    t *= 1000  # convert to ms

    # Neighborhood from electrode positions
    neighbors = electrode_neighborhoods(mea='hidens', x=x, y=y)

    Model1 = ModelDiscriminatorBakkum()
    Model1.fit(t, V, pnr_threshold=5)
    Model1.predict()
    Model2 = ModelDiscriminatorBullmann()
    Model2.fit(t, V, neighbors)
    Model2.predict()

    evaluation = Experiment(FIGURE_CULTURE).comparison_of_discriminators(FIGURE_NEURONS)

    # Plotting Frames A~E
    fig = plt.figure(figurename, figsize=(13, 10))
    if not figpath:
        fig.suptitle(figurename + ' Comparison of segmentation methods', fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.90, bottom=0.05)

    ax1 = plt.subplot(231)
    Model1.plot(ax1, xlabel=r'$\log_{10}(V_{n}/\sigma_{V})$', fontsize=14)
    adjust_position(ax1, xshrink=0.01)
    ax1.text(-0.3,450, 'Method I', size=14)
    ax1.set_ylim((0,500))
    without_spines_and_ticks(ax1)
    ax1.annotate('(fixed) threshold \n$%d\sigma_{V}$' % np.power(10, Model1.threshold),
                 xy=(Model1.threshold, 0),
                 xytext=(Model1.threshold, 200),
                 arrowprops=dict(facecolor='black', width=1),
                 size=14)
    plt.title('a', loc='left', fontsize=18)

    ax2 = plt.subplot(232)
    Model2.plot(ax2, xlabel=r'$\frac{s_{\tau}}{T/2}$', fontsize=20)
    adjust_position(ax2, xshrink=0.01)
    ax2.text(0.1,450, 'Method II', size=14)
    ax2.set_ylim((0,500))
    without_spines_and_ticks(ax2)
    ax2.annotate('(adaptive) threshold \n$s_{min}=%1.3f$ms' % Model2.threshold,
                 xy=(Model2.threshold, 0),
                 xytext=(Model2.threshold, 200),
                 arrowprops=dict(facecolor='black', width=1),
                 size=14)
    plt.title('b', loc='left', fontsize=18)

    ax3 = plt.subplot(233)
    Model1.plot_ROC(ax3, color='blue', marker='d', label = 'I')
    Model2.plot_ROC(ax3, color='black', marker='o', label = 'II')
    without_spines_and_ticks(ax3)
    ax3.plot((0,1),(0,1), 'k--', label ='chance')
    ax3.set_xlim((0,1))
    ax3.set_ylim((0,1))
    ax3.legend(loc=4, scatterpoints=1, frameon=False)
    plt.title('c', loc='left', fontsize=18)

    ax4 = plt.subplot(234)
    Model1.plot_Map(ax4, x, y)
    ax4.text(300, 300, r'I: $V_{n} > %d\sigma_{V}; \tau > \tau_{AIS}$' % np.power(10, Model1.threshold),
            bbox=dict(facecolor='white', pad=5, edgecolor='none'), size=14)
    mea_axes(ax4)
    plt.title('d', loc='left', fontsize=18)

    ax5 = plt.subplot(235)
    Model2.plot_Map(ax5, x, y)
    ax5.text(300, 300, r'II: $s_{\tau} < s_{min}; \tau > \tau_{AIS}$', bbox=dict(facecolor='white', pad=5, edgecolor='none'), size=14)
    mea_axes(ax5)
    plt.title('e', loc='left', fontsize=18)

    # Plotting Evaluation
    ax6 =plt.subplot(2,9,16)
    plot_pairwise_comparison(ax6, evaluation, 'AUC', ylim=(0, 0.5), legend=False)
    adjust_position(ax6, yshrink = 0.05)
    plt.title('f', loc='left', fontsize=18)
    ax6b =plt.subplot(2,9,17)
    plot_pairwise_comparison(ax6b, evaluation, 'TPR', ylim=(0, 1), legend=False)
    adjust_position(ax6b, xshift = 0.01, yshrink = 0.05)
    plt.title('g', loc='left', fontsize=18)
    ax6c =plt.subplot(2,9,18)
    plot_pairwise_comparison(ax6c, evaluation, 'FPR', ylim=(0, 0.02), legend=False)
    adjust_position(ax6c, xshift = 0.02, yshrink = 0.05)
    plt.title('h', loc='left', fontsize=18)

    show_or_savefig(figpath, figurename)


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))