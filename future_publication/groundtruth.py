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
    delay_old, _, _, index_AIS, axon_pnr5 = segment_axon_Bakkum(V, t, pnr_threshold=5)
    _, _, _, _, axon_pnr3 = segment_axon_Bakkum(V, t, pnr_threshold=3)



    # Segmentation according Bullmann
    neighbors = neighbors_from_electrode_positions(x, y)
    delay, mean_delay, std_delay, expected_std_delay, thr, valid_delay, index_AIS, positive_delay, axon \
        = segment_axon_verbose(t, V, neighbors)

    from future_publication.grid import HidensTransformation
    hidens = HidensTransformation(x,y)
    xs, ys, Vs = hidens.subset(V, ioffset=0, joffset=1)
    neighborss = neighbors_from_electrode_positions(xs, ys, neighborhood_radius=60)
    delays, _, std_delay, _, _, _, _, _, _  = segment_axon_verbose(t, Vs, neighborss)
    axon_2 = std_delay < thr  # using the threshold, because less data to compute histogram

    # AIS coordinates
    x_AIS = x[index_AIS]
    y_AIS = y[index_AIS]

    Vmin = np.min(V, axis=1)
    Vmins = np.min(Vs, axis=1)

    # Plotting
    fig = plt.figure('Figure X neuron %d' % neuron, figsize=(15, 11))
    fig.suptitle('Figure X. Ground truth for neuron %d' % neuron, fontsize=14, fontweight='bold')
    bbox = [200,800,1700,1200]

    max_V = np.round(np.max(np.abs(x)),-3)

    V2size = lambda x : np.abs(x)/np.sqrt(max_V) * 100

    # Map axons for Bakkum's method, high threshold
    ax1 = plt.subplot(331)
    plot_image_axon_delay_voltage(ax1, path+'axon', axon_pnr5, delay_old, Vmin, x, y, transform=V2size)
    cross_hair(ax1, x_AIS, y_AIS, color='red')
    set_axis_data(bbox)
    ax1.set_title('Method I')

    # Map axons for Bullmann's method, grid spacing ~ 20um
    ax2 = plt.subplot(332)
    ax2h = plot_image_axon_delay_voltage(ax2, path+'axon', axon, delay, Vmin, x, y, transform=V2size)
    cross_hair(ax2, x_AIS, y_AIS, color='red')
    set_axis_data(bbox)
    ax2.set_title('Method II')

    # Ground truth
    ax3 = plt.subplot(333)
    plot_neuron_image(path)
    cross_hair(ax2, x_AIS, y_AIS, color='red')
    set_axis_data(bbox)
    ax3.set_title('Groundtruth')

    # Map axons for Bakkum's method, low threshold
    ax4 = plt.subplot(334)
    plot_image_axon_delay_voltage(ax4, path+'axon', axon_pnr3, delay_old, Vmin, x, y, transform=V2size)
    cross_hair(ax4, x_AIS, y_AIS, color='red')
    set_axis_data(bbox)

    # Map axons for Bullmann's method, grid spacing ~ 40um
    ax5 = plt.subplot(335)
    plot_image_axon_delay_voltage(ax5, path+'axon', axon_2, delays, Vmins, xs, ys, transform=V2size)
    cross_hair(ax5, x_AIS, y_AIS, color='red')
    set_axis_data(bbox)

    # Colorbar for A, B, D, E
    ax6 = plt.subplot(336)
    for V in [1, 3, 10, 30, 100]:
        plt.scatter([],[],s=V2size(V), color='white', edgecolor='black', label='%d' % V)
    leg = plt.legend(loc=2, scatterpoints=1, frameon=False, title = r'$\mathsf{V_n\ [\mu V]}\ \ \ \ \ $')
    leg.get_title().set_fontsize(14)
    plt.axis('off')
    adjust_position(ax6,xshrink=0.05,yshrink=0.02,xshift=-0.04)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes("right", size="10%", pad=0.05)

    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0, vmax=2)
    cbar = mpl.colorbar.ColorbarBase(cax,
                                    norm=norm,
                                    orientation='vertical')
    cbar.set_label(r'$\mathsf{\tau_{axon}\ [ms]}$', fontsize=14)

    # Testing ground truth, reading xg, yg from the axon label file(s)
    ax7 = plt.subplot(337)
    plot_neuron_image(path+'axon')
    xg, yg = groundtruth_neuron_image(path + 'axon')

    plt.scatter(xg,yg)

    plt.show()


def set_axis_data(bbox):
    set_bbox(bbox)
    plt.xlabel(r'$\mathsf{x\ [\mu m]}$')
    plt.ylabel(r'$\mathsf{y\ [\mu m]}$')


def set_bbox(bbox):
    plt.xlim((bbox[0], bbox[1]))
    plt.ylim((bbox[2], bbox[3]))


def plot_image_axon_delay_voltage(ax, path, axon, delay, V, x, y, transform=np.abs):
    plot_neuron_image(path)
    radius = transform(V) if transform else V
    s = ax.scatter(x[axon], y[axon], s=radius[axon], c=delay[axon], marker='o', vmin=0, vmax=2, alpha=0.5)
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


def groundtruth_neuron_image(path):
    """
    Note: See plot_neuron_image
    :param path: path to tilespecs.txt
    :return: x, y: coordinates of foreground pixels (value>0)
    """
    textfilename = os.path.join(path, 'tilespecs.txt')
    tiles = pd.read_table(textfilename)
    x_list = list()
    y_list = list()
    for tile_index, tile in tiles.iterrows():
        raw_image = plt.imread(os.path.join(path, tile.filename))
        heigth, width = np.shape(raw_image)
        i, j = np.where(raw_image)
        x_list.append((tile.xend - tile.xstart) / width * j + tile.xstart)
        y_list.append((tile.yend - tile.ystart) / heigth * i + tile.ystart)
    x = np.hstack(x_list)
    y = np.hstack(y_list)
    return x, y


def distanceBetweenCurves(C1, C2):
    """
    From http://stackoverflow.com/questions/13692801/distance-matrix-of-curves-in-python
    Note: According to https://en.wikipedia.org/wiki/Hausdorff_distance this is defined
    as max(H1, H2) and not (H1 + H2) / 2.
    :param C1, C2: to curves as their points
    :return:
    """
    D = scipy.spatial.distance.cdist(C1, C2, 'euclidean')

    #none symmetric Hausdorff distances
    H1 = np.max(np.min(D, axis=1))
    H2 = np.max(np.min(D, axis=0))

    return (H1 + H2) / 2.


if __name__ == "__main__":
    testing_load_traces(1544)
    # testing_load_traces(1536)
