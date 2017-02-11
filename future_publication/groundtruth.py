import logging
logging.basicConfig(level=logging.DEBUG)

import os as os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


from hana.plotting import annotate_x_bar, set_axis_hidens
from hana.recording import half_peak_width, peak_peak_width, peak_peak_domain, DELAY_EPSILON, neighborhood, \
    electrode_neighborhoods, load_traces, load_positions
from hana.segmentation import segment_axon_verbose, restrict_to_compartment, find_AIS
from publication.plotting import FIGURE_NEURON_FILE, without_spines_and_ticks, cross_hair, \
    legend_without_multiple_labels, label_subplot, plot_traces_and_delays, adjust_position, voltage_color_bar
from publication.comparison import segment_axon_Bakkum



# Testing code

def testing_load_traces(path):

    # Get traces
    V, t, x, y, trigger, neuron = load_traces(path+'.h5')
    if trigger<0:  # may added to load_traces with trigger>-1 as condition
        trigger = find_AIS(V)

    t = t/20 * 1000  # convert to ms TODO Fix factor 20

    # Segmentation according Bakkum
    delay_old, _, _, index_AIS, axon_old = segment_axon_Bakkum(V, t, pnr_threshold=5)

    # Segmentation according Bullmann
    neighbors = neighbors_from_electrode_positions(x, y)
    delay, mean_delay, std_delay, expected_std_delay, thr, valid_delay, index_AIS, positive_delay, axon \
        = segment_axon_verbose(t, V, neighbors)

    # AIS coordinates
    x_AIS = x[index_AIS]
    y_AIS = y[index_AIS]

    V = np.min(V, axis=1)

    # Plotting
    fig = plt.figure('Figure X', figsize=(13, 7))
    fig.suptitle('Figure X. Ground truth', fontsize=14, fontweight='bold')

    # Map axons
    ax4 = plt.subplot(121)
    plot_neuron_image(path)
    ax4.scatter(x, y, s=axon_old * 40, c='blue', marker='o', alpha=0.5)
    cross_hair(ax4, x_AIS, y_AIS, color='red')
    set_axis_hidens(ax4)

    ax5 = plt.subplot(122)
    plot_neuron_image(path)
    ax5.scatter(x, y, s=axon * 40, c='yellow', marker='o', alpha=0.5)
    cross_hair(ax5, x_AIS, y_AIS, color='red')

    set_axis_hidens(ax5)

    plt.show()


def neighbors_from_electrode_positions(x, y, neighborhood_radius = 20):
    # only a subset of electrodes used TODO: generalize and merge electrode_neighborhoods
    from scipy.spatial.distance import squareform, pdist

    pos_as_array = np.asarray(zip(x, y))
    distances = squareform(pdist(pos_as_array, metric='euclidean'))
    neighbors = distances < neighborhood_radius
    return neighbors


def plot_neuron_image(path, transform=np.sqrt, cmap='gray_r'):
    """

    :param path: path to tilespecs.txt
    :param transform: image transformation; default sqrt, could be None as well
    :param cmap: colormap; default: inverted gray scale
    """
    textfilename = os.path.join(path,'tilespecs.txt')
    tiles = pd.read_table(textfilename)
    for tile_index, tile in tiles.iterrows():
        raw_image = plt.imread(os.path.join(path,tile.filename))
        image = transform(raw_image) if transform else raw_image
        plt.imshow(image,
                   cmap=cmap,
                   extent=[tile.xstart, tile.xend, tile.yend, tile.ystart])
        # NOTE: yend and ystart are reverted due to hidens coordinate system TODO: Maybe fix





testing_load_traces('data/groundtruth/neuron1544')
testing_load_traces('data/groundtruth/neuron1536')


