import logging

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from hana.plotting import mea_axes
from hana.recording import electrode_neighborhoods, load_traces
from publication.comparison import ModelDiscriminatorBakkum, ModelDiscriminatorBullmann
from publication.plotting import show_or_savefig, FIGURE_NEURON_FILE_FORMAT, FIGURE_NEURONS, label_subplot, adjust_position, \
    plot_pairwise_comparison, without_spines_and_ticks

logging.basicConfig(level=logging.DEBUG)
DISCRIMINATOR_EVALUATION_FILENAME = 'temp/comparison_of_discriminators.csv'


def make_figure(figurename, figpath=None):

    # Load example data
    neuron = 5  # other neurons 5, 10, 11, 20, 25, 2, 31, 41
    filename = FIGURE_NEURON_FILE_FORMAT % neuron

    V, t, x, y, trigger, neuron = load_traces(filename)
    t *= 1000  # convert to ms

    # Neighborhood from electrode positions
    neighbors = electrode_neighborhoods(mea='hidens', x=x, y=y)

    Model1 = ModelDiscriminatorBakkum()
    Model1.fit(t, V, pnr_threshold=5)
    Model1.predict()
    Model2 = ModelDiscriminatorBullmann()
    Model2.fit(t, V, neighbors)
    Model2.predict()

    evaluation = compare_discriminators()

    # Summary
    print (Model1.summary(subject=filename, method='Bakkum'))
    print (Model2.summary(subject=filename, method='Bullmann'))

    # Plotting Frames A~E
    fig = plt.figure(figurename, figsize=(16, 10))
    fig.suptitle(figurename + ' Comparison of segmentation methods', fontsize=14, fontweight='bold')

    ax1 = plt.subplot(231)
    Model1.plot(ax1, xlabel=r'$\log_{10}(V_{n}/\sigma_{V})$', fontsize=14)
    adjust_position(ax1, xshrink=0.01)
    ax1.text(-0.3,450, 'I', size=14)
    ax1.set_ylim((0,500))
    without_spines_and_ticks(ax1)
    ax1.annotate('(fixed) threshold \n$%d\sigma_{V}$' % np.power(10, Model1.threshold),
                 xy=(Model1.threshold, 0),
                 xytext=(Model1.threshold, 200),
                 arrowprops=dict(facecolor='black', width=1),
                 size=14)

    ax2 = plt.subplot(232)
    Model2.plot(ax2, xlabel=r'$s_{\tau}$ [ms]', fontsize=14)
    adjust_position(ax2, xshrink=0.01)
    ax2.text(0.3,450, 'II', size=14)
    ax2.set_ylim((0,500))
    without_spines_and_ticks(ax2)
    ax2.annotate('(adaptive) threshold \n$s_{min}=%1.3f$ms' % Model2.threshold,
                 xy=(Model2.threshold, 0),
                 xytext=(Model2.threshold, 200),
                 arrowprops=dict(facecolor='black', width=1),
                 size=14)

    ax3 = plt.subplot(233)
    Model1.plot_ROC(ax3, color='blue', marker='x', label = 'I')
    Model2.plot_ROC(ax3, color='black', marker='o', label = 'II')
    without_spines_and_ticks(ax3)
    ax3.plot((0,1),(0,1), 'k--', label ='chance')
    ax3.set_xlim((0,1))
    ax3.set_ylim((0,1))
    ax3.legend(loc=4, scatterpoints=1, frameon=False)

    ax4 = plt.subplot(234)
    Model1.plot_Map(ax4, x, y)
    ax4.text(300, 300, r'I: $V_{n} > %d\sigma_{V}; \tau > \tau_{AIS}$' % np.power(10, Model1.threshold),
            bbox=dict(facecolor='white', pad=5, edgecolor='none'), size=14)
    mea_axes(ax4)

    ax5 = plt.subplot(235)
    Model2.plot_Map(ax5, x, y)
    ax5.text(300, 300, r'II: $s_{\tau} < s_{min}; \tau > \tau_{AIS}$', bbox=dict(facecolor='white', pad=5, edgecolor='none'), size=14)
    mea_axes(ax5)

    # Plotting Evaluation
    ax6 =plt.subplot(2,9,16)
    plot_pairwise_comparison(ax6, evaluation, 'AUC', ylim=(0, 0.5), legend=False)
    ax6b =plt.subplot(2,9,17)
    plot_pairwise_comparison(ax6b, evaluation, 'TPR', ylim=(0, 1), legend=False)
    adjust_position(ax6b, xshift = 0.01)
    ax6c =plt.subplot(2,9,18)
    plot_pairwise_comparison(ax6c, evaluation, 'FPR', ylim=(0, 0.01), legend=False)
    adjust_position(ax6c, xshift = 0.02)

    label_subplot(ax1, 'A', xoffset=-0.05, yoffset=-0.01)
    label_subplot(ax2, 'B', xoffset=-0.05, yoffset=-0.01)
    label_subplot(ax3, 'C', xoffset=-0.04, yoffset=-0.01)
    label_subplot(ax4, 'D', xoffset=-0.03, yoffset=-0.01)
    label_subplot(ax5, 'E', xoffset=-0.03, yoffset=-0.01)
    label_subplot(ax6, 'F', xoffset=-0.04, yoffset=-0.01)

    show_or_savefig(figpath, figurename)


def compare_discriminators():
    """
    Evaluating both models for all neurons, load from csv if already exist.
    :return: pandas data frame
    """

    if os.path.isfile(DISCRIMINATOR_EVALUATION_FILENAME):
        data = pd.DataFrame.from_csv(DISCRIMINATOR_EVALUATION_FILENAME)

    else:
        Model1 = ModelDiscriminatorBakkum()
        Model2 = ModelDiscriminatorBullmann()

        # Load electrode coordinates
        neighbors = electrode_neighborhoods(mea='hidens')

        evaluations = list()
        for neuron in FIGURE_NEURONS:
            filename = FIGURE_NEURON_FILE_FORMAT % neuron
            V, t, x, y, trigger, neuron = load_traces(filename)
            t *= 1000  # convert to ms

            Model1.fit(t, V, pnr_threshold=5)
            Model1.predict()
            Model2.fit(t, V, neighbors)
            Model2.predict()

            evaluations.append(Model1.summary(subject='%d' % neuron, method='I'))
            evaluations.append(Model2.summary(subject='%d' % neuron, method='II'))

        data = pd.DataFrame(evaluations)
        data.to_csv(DISCRIMINATOR_EVALUATION_FILENAME)

    return data


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))