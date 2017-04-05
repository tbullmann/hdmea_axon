import pickle

import numpy as np
from matplotlib import pyplot as plt

from publication.data import Experiment, FIGURE_CULTURE
from publication.plotting import show_or_savefig, plot_parameter_dependency, TEMPORARY_PICKELED_NETWORKS, compare_networks, \
    format_parameter_plot


def make_figure(figurename, figpath=None, networks_pickel_name=TEMPORARY_PICKELED_NETWORKS):

    structural_strengths, structural_delays, functional_strengths, functional_delays \
        = Experiment(FIGURE_CULTURE).networks()

    # Making figure
    fig = plt.figure(figurename, figsize=(16, 16))
    fig.suptitle(figurename + ' Compare structural and functional connectivity', fontsize=14, fontweight='bold')

    structural_thresholds, functional_thresholds, intersection_size, structural_index, jaccard_index, functional_index\
        = compare_networks(structural_strengths, functional_strengths, scale='log')

    ax1 = plt.subplot(2,2,1)
    plot_parameter_dependency(ax1, intersection_size, structural_thresholds, functional_thresholds,
                              fmt='%d', levels=(1,2,5,10,20,50,100,150))
    format_parameter_plot(ax1)
    ax1.set_title(r'intersection size $|F \cup S|$')

    ax2 = plt.subplot(2,2,2)
    plot_parameter_dependency(ax2, structural_index, structural_thresholds, functional_thresholds,
                              fmt='%1.1f', levels=np.linspace(0,1,11))
    format_parameter_plot(ax2)
    ax2.set_title(r'Structural index ${|F \cup S|}/{|S|}$')

    ax3 = plt.subplot(2,2,3)
    plot_parameter_dependency(ax3, jaccard_index, structural_thresholds, functional_thresholds,
                              fmt='%1.1f', levels=np.linspace(0,1,11))
    format_parameter_plot(ax3)
    ax3.set_title(r'Jaccard index ${|F \cup S|}/{|F \cap S|}$')

    ax4 = plt.subplot(2,2,4)
    plot_parameter_dependency(ax4, functional_index, structural_thresholds, functional_thresholds,
                              fmt='%1.1f', levels=np.linspace(0,1,11))
    format_parameter_plot(ax4)
    ax4.set_title(r'Functional index ${|F \cup S|}/{|F|}$')

    show_or_savefig(figpath, figurename)


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))
