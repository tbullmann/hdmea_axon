import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import logging

from hana.plotting import annotate_x_bar, mea_axes
from hana.recording import DELAY_EPSILON, neighborhood, electrode_neighborhoods
from hana.segmentation import segment_axon_verbose, restrict_to_compartment

from publication.data import FIGURE_CULTURE, FIGURE_NEURON
from publication.experiment import Experiment
from publication.plotting import show_or_savefig, cross_hair, adjust_position, make_axes_locatable
from publication.figure_groundtruth import plot_image_axon_delay_voltage

logging.basicConfig(level=logging.DEBUG)


def make_figure(figurename, figpath=None):

    # Load electrode coordinates
    neighbors = electrode_neighborhoods(mea='hidens')

    # Load example data
    V, t, x, y, trigger, neuron = Experiment(FIGURE_CULTURE).traces(FIGURE_NEURON)
    t *= 1000  # convert to ms

    # Load background images
    images = Experiment(FIGURE_CULTURE).images(FIGURE_NEURON, type='axon')


    # Verbose axon segmentation function
    delay, mean_delay, std_delay, expected_std_delay, thr, valid_delay, index_AIS, positive_delay, axon \
        = segment_axon_verbose(t, V, neighbors)

    max_axon_delay = np.around(np.nanmax(restrict_to_compartment(mean_delay, axon)), decimals=1)
    print max_axon_delay

    # Plot figure
    fig = plt.figure(figurename, figsize=(13, 7))
    ax = plt.subplot(121)

    # Map axons for Bullmann's method
    V2size = lambda x: np.sqrt(np.abs(x)) * 5
    images.plot(alpha=0.5)
    radius = V2size(V)
    ax.scatter(x[axon], y[axon], s=radius[axon], c=delay[axon],
               marker='o', edgecolor='none',
               cmap=plt.cm.hsv, vmin=0, vmax=max_axon_delay, alpha=1)

    # cross_hair(ax, x[index_AIS], y[index_AIS], color='red')
    mea_axes(ax)
    # mea_axes(ax, style='axes')

    # Colorbar and Size legend for A, B, D, E
    ax6 = plt.subplot(322)
    for legend_V in [1, 3, 10, 30, 100]:
        plt.scatter([], [], s=V2size(legend_V), color='gray', edgecolor='none', label='%d' % legend_V)
    leg = plt.legend(loc=2, scatterpoints=1, frameon=False, title=r'$\mathsf{V_n\ [\mu V]}\ \ \ \ \ $')
    leg.get_title().set_fontsize(14)
    plt.axis('off')
    adjust_position(ax6, xshrink=0.05, yshrink=0.02, xshift=-0.04)
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes("right", size="10%", pad=0.05)

    norm = mpl.colors.Normalize(vmin=0, vmax=max_axon_delay)
    cbar = mpl.colorbar.ColorbarBase(cax,
                                     cmap=plt.cm.hsv,
                                     norm=norm,
                                     orientation='vertical')
    cbar.set_label(r'$\mathsf{\tau_{axon}\ [ms]}$', fontsize=14)


    show_or_savefig(figpath, figurename)

    # Plot traces over high res background images
    images = Experiment(FIGURE_CULTURE).images(FIGURE_NEURON, type='axonhires')
    plot_full_footprint(V, axon, delay, max_axon_delay, images, figpath, figurename+"_full_arbor", index_AIS, t, x, y)


def plot_full_footprint(V, axon, delay, max_axon_delay, images, figpath, figurename, index_AIS, t, x, y):
    # Making figure
    fig = plt.figure(figurename, figsize=(13, 13))
    if not figpath:
        fig.suptitle(figurename + ' Segmentation of the axon based on negative peak at neighboring electrodes',
                     fontsize=14, fontweight='bold')

    ax = plt.subplot(111)

    # plot background picture
    images.plot(alpha=0.5)

    # plot traces
    background_color = 'silver'
    time_factor = 1.8
    voltage_factor = - 0.5
    dot_voltage = 5
    delay_factor = 1. / max_axon_delay
    cmap = plt.cm.hsv
    for el in range(0, len(V)):
    # for el in range(3000, 3100):
        print('Trace at electrode %d plotted' % el)
        el_time = time_factor * t + x[el]
        el_voltage = voltage_factor * V[el] + y[el]
        el_color = cmap(delay_factor * delay[el]) if axon[el] else background_color

        # plot trace
        ax.plot(el_time, el_voltage, color=el_color, linewidth=0.5)

        # add marker for t=0
        # ax.scatter(x[el], y[el] + voltage_factor * dot_voltage, 1, color=background_color, marker='.')
    # cross_hair(ax, x[index_AIS], y[index_AIS])
    mea_axes(ax)
    # ax.set_xlim((180,1650))
    # ax.set_ylim((450,2100))

    show_or_savefig(figpath, figurename, dpi=1200)


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))