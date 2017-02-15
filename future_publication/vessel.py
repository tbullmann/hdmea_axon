import logging


logging.basicConfig(level=logging.DEBUG)

import os as os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize

from hana.recording import load_traces
from hana.segmentation import segment_axon_verbose, find_AIS
from publication.plotting import cross_hair
from _frangi import frangi
from flow import interpolate
from groundtruth import plot_neuron_image, neighbors_from_electrode_positions, set_bbox

# Testing code

def with_continous_time(neuron, compute_anyway=False):

    V, t, x, y, x_AIS, y_AIS, axon, delay, path = load_data(neuron)

    offset = (np.where(t==0))[0]
    step = 1
    delta = 3

    skeleton_pickel_name = 'temp/skeletons.p'




    if compute_anyway or not os.path.isfile(skeleton_pickel_name):

        skeletons = list()
        for i in range(0, 40):
            index = i * step + offset
            logging.info('frame %d' % i)

            V_n = abs(np.min(V[:,index:index+delta], axis=1))
            grid_x, grid_y, _, _, skeleton = AxonSkeleton(V_n, x, y, support=axon)

            skeletons.append(skeleton)
            plt.imsave('temp/skeletons/frame%03d.tif' % i, skeleton)

        skeletonstack = np.stack(skeletons, axis=0)
        pickle.dump( ( grid_x, grid_y, skeletonstack ), open(skeleton_pickel_name, 'wb'))

    else:

        grid_x, grid_y, skeletonstack = pickle.load(open(skeleton_pickel_name, 'rb'))



    # Making figure
    fig = plt.figure('Figure step time', figsize=(18, 14))
    fig.suptitle('Segmentation of the axon based on vesselness (step time)', fontsize=14,
                 fontweight='bold')
    h=3
    w=4
    for i in range(0, h * w):
        index = i * step + offset

        bbox = [np.amin(grid_x), np.amax(grid_x), np.amax(grid_y), np.amin(grid_y)]  # for grid

        plt.subplot(h,w,i+1)
        plot_neuron_image(path)
        plt.imshow(skeletonstack[i].T, cmap='CMRmap', alpha=0.5,
                   extent=[np.amin(grid_x), np.amax(grid_x), np.amax(grid_y), np.amin(grid_y)])
        plt.title('%1.3f~%1.3f ms' % (t[index], t[index+delta]) )
        plt.axis('off')
        set_bbox(bbox)

    plt.show()


def with_collapsed_time(neuron):

    V, t, x, y, x_AIS, y_AIS, axon, delay, path = load_data(neuron)

    V_n = abs(np.min(V, axis=1))
    grid_x, grid_y, grid_V, vesselness, skeleton = AxonSkeleton(V_n, x, y, support=axon)

    # Making figure
    fig = plt.figure('Figure collapsed time', figsize=(18, 14))
    fig.suptitle('Segmentation of the axon based on vesselness (collapsed time)', fontsize=14,
                 fontweight='bold')

    bbox = [np.amin(grid_x), np.amax(grid_x), np.amax(grid_y), np.amin(grid_y)]  # for grid

    ax1 = plt.subplot(221)
    plot_neuron_image(path)
    ax1.scatter(x[axon], y[axon], s=20, c=delay[axon], marker='o', )
    cross_hair(ax1, x_AIS, y_AIS, color='red')
    ax1.set_title('Delays')
    set_bbox(bbox)

    ax2 = plt.subplot(222)
    plot_neuron_image(path)
    plt.imshow(grid_V.T, cmap='CMRmap', alpha=0.5, extent=bbox)
    ax2.set_title('Voltage')
    set_bbox(bbox)

    ax3 = plt.subplot(223)
    plot_neuron_image(path)
    plt.imshow(vesselness.T, cmap='CMRmap', alpha=0.5, extent=bbox)
    ax3.set_title('Voltage ridge')
    set_bbox(bbox)

    ax4 = plt.subplot(224)
    plot_neuron_image(path)
    plt.imshow(skeleton.T, cmap='CMRmap', alpha=0.5,
               extent=bbox)
    ax4.set_title('Axon')
    set_bbox(bbox)

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
    # with_continous_time(1544)

