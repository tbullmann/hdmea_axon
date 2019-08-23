import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import logging

from hana.plotting import annotate_x_bar, mea_axes
from hana.recording import DELAY_EPSILON, neighborhood, electrode_neighborhoods
from hana.segmentation import segment_axon_verbose, restrict_to_compartment

from data import FIGURE_CULTURE, FIGURE_NEURON
from experiment import AxonExperiment
from plotting import show_or_savefig, cross_hair, adjust_position, make_axes_locatable
from figure_groundtruth import plot_image_axon_delay_voltage

logging.basicConfig(level=logging.DEBUG)


def make_figure(figurename, figpath=None):

    # Load electrode coordinates
    neighbors = electrode_neighborhoods(mea='hidens')

    # Load example data
    V, t, x, y, trigger, neuron = AxonExperiment(FIGURE_CULTURE).traces(FIGURE_NEURON)
    t *= 1000  # convert to ms

    # Load background images
    images = AxonExperiment(FIGURE_CULTURE).images(FIGURE_NEURON, type='axon')


    # Verbose axon segmentation function
    delay, mean_delay, std_delay, expected_std_delay, thr, valid_delay, index_AIS, positive_delay, axon \
        = segment_axon_verbose(t, V, neighbors)

    # Plot figure
    fig = plt.figure(figurename, figsize=(13, 7))

    # Map activity
    triggers, AIS, delays, positive_peak = AxonExperiment(FIGURE_CULTURE).compartments()
    map_data = AxonExperiment(FIGURE_CULTURE).event_map()
    ax1 = plt.subplot(121)
    ax1.scatter(map_data['x'] , map_data['y'],
                # s=10,
                s=0.75 * np.sqrt(map_data['count']),
                c=-map_data['median_neg_peak'],
                marker='o', edgecolor='none',
                cmap=plt.cm.Blues, vmin=0, vmax=100, alpha=1)
    # trigger_indicies = np.unique(triggers.values()).astype(np.int16)-1
    # ax1.scatter(x[trigger_indicies],y[trigger_indicies], 10, color='red', marker='x')
    # trigger_indicies = Experiment(FIGURE_CULTURE).fixed_electrodes()['el']-1
    # ax1.scatter(x[trigger_indicies], y[trigger_indicies], 10, color='green', marker='+')
    print(max(map_data['count']))
    print(min(map_data['median_neg_peak']))
    # cross_hair(ax1, x[index_AIS], y[index_AIS], color='red')
    mea_axes(ax1)

    # Map axons for Bullmann's method
    ax2 = plt.subplot(122)
    max_axon_delay = np.around(np.nanmax(restrict_to_compartment(mean_delay, axon)), decimals=1)
    V2size = lambda x: np.sqrt(np.abs(x)) * 5
    images.plot(alpha=0.5)
    radius = V2size(V)
    ax2.scatter(x[axon], y[axon], s=radius[axon], c=delay[axon],
               marker='o', edgecolor='none',
               cmap=plt.cm.hsv, vmin=0, vmax=max_axon_delay, alpha=1)

    # cross_hair(ax2, x[index_AIS], y[index_AIS], color='red')
    mea_axes(ax2)

    # # Colorbar and Size legend for A, B, D, E
    # ax6 = plt.subplot(322)
    # for legend_V in [1, 3, 10, 30, 100]:
    #     plt.scatter([], [], s=V2size(legend_V), color='gray', edgecolor='none', label='%d' % legend_V)
    # leg = plt.legend(loc=2, scatterpoints=1, frameon=False, title=r'$\mathsf{V_n\ [\mu V]}\ \ \ \ \ $')
    # leg.get_title().set_fontsize(14)
    # plt.axis('off')
    # adjust_position(ax6, xshrink=0.05, yshrink=0.02, xshift=-0.04)
    # divider = make_axes_locatable(ax6)
    # cax = divider.append_axes("right", size="10%", pad=0.05)
    #
    # norm = mpl.colors.Normalize(vmin=0, vmax=max_axon_delay)
    # cbar = mpl.colorbar.ColorbarBase(cax,
    #                                  cmap=plt.cm.hsv,
    #                                  norm=norm,
    #                                  orientation='vertical')
    # cbar.set_label(r'$\mathsf{\tau_{axon}\ [ms]}$', fontsize=14)

    show_or_savefig(figpath, figurename)
    plt.close()

    # Plot traces over high res background images
    images = AxonExperiment(FIGURE_CULTURE).images(FIGURE_NEURON, type='axonhires')
    fig = plt.figure(figurename, figsize=(13, 13))
    ax = plt.subplot(111)
    images.plot(alpha=0.5)
    plot_traces(ax, x, y, axon, delay, t, V, max_axon_delay)
    mea_axes(ax)

    show_or_savefig(figpath, figurename+"_full_arbor", dpi=1200)
    plt.close()


def plot_traces(ax, x, y, axon, delay, t, V,
                max_axon_delay=None, time_factor = 1.8, voltage_factor = - 0.5, background_color = 'silver'):
    if not max_axon_delay:
        max_axon_delay = restrict_to_compartment(axon, delay)
    delay_factor = 1. / max_axon_delay
    cmap = plt.cm.hsv
    for el in range(0, len(V)):
        print('Trace at electrode %d plotted' % el)
        el_time = time_factor * t + x[el]
        el_voltage = voltage_factor * V[el] + y[el]
        el_color = cmap(delay_factor * delay[el]) if axon[el] else background_color
        ax.plot(el_time, el_voltage, color=el_color, linewidth=0.5)


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))