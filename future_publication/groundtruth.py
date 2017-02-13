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

def testing_load_traces(neuron):

    path ='data/groundtruth/neuron%d' % neuron

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
    fig = plt.figure('Figure X neuron %d' % neuron, figsize=(13, 7))
    fig.suptitle('Figure X. Ground truth for neuron %d' % neuron, fontsize=14, fontweight='bold')

    max_V = np.round(np.max(np.abs(x)),-3)
    print max_V
    print max(delay)
    V2size = lambda x : np.sqrt(np.abs(x))/np.sqrt(max_V) * 100

    # Map axons
    ax4 = plt.subplot(131)
    plot_image_axon_delay_voltage(ax4, path, axon_old, delay_old, V, x, y, transform=V2size)
    cross_hair(ax4, x_AIS, y_AIS, color='red')
    set_axis_hidens(ax4)
    ax4.set_title('Method I')

    ax5 = plt.subplot(132)
    ax5h = plot_image_axon_delay_voltage(ax5, path, axon, delay, V, x, y, transform=V2size)
    cross_hair(ax5, x_AIS, y_AIS, color='red')
    set_axis_hidens(ax5)
    ax5.set_title('Method II')

    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # divider = make_axes_locatable(ax5)
    # cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(ax5h)
    cbar.set_label(r'$\mathsf{\tau_{axon}\ [ms]}$', fontsize=14)

    for V in [1, 10, 100, 1000]:
        plt.scatter([],[],s=V2size(V), color='white', edgecolor='black', label='%d' % V)
    # leg = plt.legend(scatterpoints=1, frameon=False, loc = 'upper left', bbox_to_anchor = (1, 1),  title = r'$\mathsf{V_n\ [\mu V]}$')
    leg = plt.legend(scatterpoints=1, frameon=False, title = r'$\mathsf{V_n\ [\mu V]}\ \ \ \ \ $')
    leg.get_title().set_fontsize(14)



    plt.show()


def plot_image_axon_delay_voltage(ax, path, axon, delay, V, x, y, transform=np.abs):
    plot_neuron_image(path)
    radius = transform(V) if transform else V
    s = ax.scatter(x[axon], y[axon], s=radius[axon], c=delay[axon], marker='o', alpha=0.5)
    return s


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


def distanceBetweenCurves(C1, C2):
    """
    From http://stackoverflow.com/questions/13692801/distance-matrix-of-curves-in-python
    Note: According to https://en.wikipedia.org/wiki/Hausdorff_distance this is defined
    as max(H1, H2) and not (H1 + H2) / 2.
    :param C1:
    :param C2:
    :return:
    """
    D = scipy.spatial.distance.cdist(C1, C2, 'euclidean')

    #none symmetric Hausdorff distances
    H1 = np.max(np.min(D, axis=1))
    H2 = np.max(np.min(D, axis=0))

    return (H1 + H2) / 2.



testing_load_traces(1544)
# testing_load_traces(1536)


