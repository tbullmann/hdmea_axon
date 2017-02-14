import logging
logging.basicConfig(level=logging.DEBUG)

import os as os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize

from hana.plotting import annotate_x_bar, set_axis_hidens
from hana.recording import half_peak_width, peak_peak_width, peak_peak_domain, DELAY_EPSILON, neighborhood, \
    electrode_neighborhoods, load_traces, load_positions
from hana.segmentation import segment_axon_verbose, restrict_to_compartment, find_AIS
from publication.plotting import FIGURE_NEURON_FILE, without_spines_and_ticks, cross_hair, \
    legend_without_multiple_labels, label_subplot, plot_traces_and_delays, adjust_position, voltage_color_bar
from publication.comparison import segment_axon_Bakkum
from _frangi import frangi
from flow import interpolate
from groundtruth import plot_neuron_image, neighbors_from_electrode_positions

# Testing code

def with_continous_time(neuron):

    V, t, x, y, x_AIS, y_AIS, axon, delay, path = load_data(neuron)

    offset = (np.where(t==0))[0]
    step = 2
    delta = 4


    # Making figure
    fig = plt.figure('Figure step time', figsize=(18, 14))
    fig.suptitle('Segmentation of the axon based on vesselness (step time)', fontsize=14,
                 fontweight='bold')

    h=4
    w=5
    for i in range(0, h*w):
        index = i * step + offset
        logging.info('frame %d' % i)

        V_n = abs(np.min(V[:,index:index+delta], axis=1))
        grid_x, grid_y, grid_V, vesselness, skeleton = AxonSkeleton(V_n, x, y, support=axon)

        plt.subplot(h,w,i+1)
        plot_neuron_image(path)
        plt.imshow(skeleton.T, cmap='CMRmap', alpha=0.5,
                   extent=[np.amin(grid_x), np.amax(grid_x), np.amax(grid_y), np.amin(grid_y)])
        plt.title('%1.3f~%1.3f ms' % (t[index], t[index+delta]) )
        plt.axis('off')

    plt.show()


def with_collapsed_time(neuron):

    V, t, x, y, x_AIS, y_AIS, axon, delay, path = load_data(neuron)

    V_n = abs(np.min(V, axis=1))
    grid_x, grid_y, grid_V, vesselness, skeleton = AxonSkeleton(V_n, x, y, support=axon)

    # Making figure
    fig = plt.figure('Figure collapsed time', figsize=(18, 14))
    fig.suptitle('Segmentation of the axon based on vesselness (collapsed time)', fontsize=14,
                 fontweight='bold')

    ax1 = plt.subplot(221)
    plot_neuron_image(path)
    ax1.scatter(x[axon], y[axon], s=20, c=delay[axon], marker='o', )
    cross_hair(ax1, x_AIS, y_AIS, color='red')
    # set_axis_hidens(ax1)
    ax1.set_title('Image+Voltage')

    ax2 = plt.subplot(222)
    plot_neuron_image(path)
    plt.imshow(grid_V.T, cmap='CMRmap', alpha=0.5, extent=[np.amin(grid_x), np.amax(grid_x), np.amax(grid_y), np.amin(grid_y)])
    ax1.set_title('Interpolated Voltage')

    ax3 = plt.subplot(223)
    plot_neuron_image(path)
    plt.imshow(vesselness.T, cmap='CMRmap', alpha=0.5, extent=[np.amin(grid_x), np.amax(grid_x), np.amax(grid_y), np.amin(grid_y)])
    ax3.set_title('Vesselness')

    ax4 = plt.subplot(224)
    plot_neuron_image(path)
    plt.imshow(skeleton.T, cmap='CMRmap', alpha=0.5,
               extent=[np.amin(grid_x), np.amax(grid_x), np.amax(grid_y), np.amin(grid_y)])
    ax4.set_title('Skeleton')

    plt.show()


def load_data(neuron):
    path = 'data/groundtruth/neuron%d' % neuron
    # Get traces
    V, t, x, y, trigger, neuron = load_traces(path + '.h5')
    if trigger < 0:  # may added to load_traces with trigger>-1 as condition
        trigger = find_AIS(V)
    t = t / 20 * 1000  # convert to ms TODO Fix factor 20
    # Segmentation according Bullmann
    neighbors = neighbors_from_electrode_positions(x, y)
    delay, mean_delay, std_delay, expected_std_delay, thr, valid_delay, index_AIS, positive_delay, axon \
        = segment_axon_verbose(t, V, neighbors)
    # AIS coordinates
    x_AIS = x[index_AIS]
    y_AIS = y[index_AIS]
    return V, t, x,  y, x_AIS, y_AIS, axon, delay, path


def AxonSkeleton(V_n, x, y, support='everywhere'):
    if not support=='everywhere':
        not_axon = np.logical_not(support)
        V_n[not_axon] = 0
    grid_x, grid_y, grid_V = interpolate(x, y, V_n, xspacing=5, yspacing=5)
    grid_V[np.isnan(grid_V)] = 0
    vesselness = frangi(-grid_V, scale_range=(2, 6), scale_step=0.5)
    skeleton = skeletonize(vesselness > 0.01)
    return grid_x, grid_y, grid_V, vesselness, skeleton


if __name__ == "__main__":
    with_collapsed_time(1544)  #1536
    with_continous_time(1544)

