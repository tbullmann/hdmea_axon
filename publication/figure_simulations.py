from __future__ import division

import logging


import matplotlib.pyplot as plt

import os

from publication.simulation import Simulation
from publication.plotting import show_or_savefig
from publication.figure_graphs import plot_scalar_measures, plot_histograms
logging.basicConfig(level=logging.DEBUG)



def make_figure(figurename, figpath=None):

    # Making figure
    fig = plt.figure(figurename, figsize=(13, 8))
    if not figpath:
        fig.suptitle(figurename + '    Compare simulated networks', fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.1)

    plot_scalar_measures(data=Simulation, all_original=False)

    plot_histograms(data=Simulation, all_original=False)

    show_or_savefig(figpath, figurename)


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))
