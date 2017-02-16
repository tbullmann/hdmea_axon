import logging

import os
import numpy as np
from scipy.spatial.distance import squareform, pdist
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hana.plotting import mea_axes
from hana.recording import load_traces
from hana.segmentation import segment_axon_verbose, find_AIS
from publication.plotting import cross_hair, label_subplot, adjust_position, without_spines_and_ticks
from publication.comparison import segment_axon_Bakkum, ImageIterator, distanceBetweenCurves

logging.basicConfig(level=logging.DEBUG)


def make_figure(figurename, neuron=1544):  # TODO add ground truth for neuron 1536

    path ='data/groundtruth/neuron%d' % neuron

    # Get traces
    V, t, x, y, trigger, neuron = load_traces(path+'.h5')
    if trigger<0:  # may added to load_traces with trigger>-1 as condition
        trigger = find_AIS(V)

    t = t/20 * 1000  # convert to ms TODO Fix factor 20 in Matlab export

    # AIS coordinates
    index_AIS = find_AIS(V)
    x_AIS = x[index_AIS]
    y_AIS = y[index_AIS]

    # Segmentation according Bakkum
    axon_Bakkum = dict()
    for pnr in (2, 3, 4, 5):
        delay, _, _, _, axon = segment_axon_Bakkum(V, t, pnr_threshold=pnr)
        axon_Bakkum[pnr] = axon
    delay_Bakkum = delay
    Vmin_Bakkum = np.min(V, axis=1)

    # Segmentation according Bullmann
    from hana.grid import HidensTransformation
    axon_Bullmann = dict()
    delay_Bullmann = dict()
    Vmin_Bullmann = dict()
    x_Bullmann = dict()
    y_Bullmann = dict()
    hidens = HidensTransformation(x,y)
    for period in (1,2,3):
        if period>1:
            xs, ys, Vs = hidens.subset(V, period=period, ioffset=0, joffset=1)
        else:
            xs, ys, Vs = x, y, V
        neighbors = neighbors_from_electrode_positions(xs, ys, neighborhood_radius=20*period)
        delay, _, std_delay, _, thr, _, _, _, axon = segment_axon_verbose(t, Vs, neighbors)
        if period==1:
            keep_thr = thr
        else:
            axon = std_delay < keep_thr  # using the threshold, because less data to compute histogram
        axon_Bullmann[period] = axon
        delay_Bullmann[period] = delay
        Vmin_Bullmann[period] = np.min(Vs, axis=1)
        x_Bullmann[period] = xs
        y_Bullmann[period] = ys

    # Plotting
    fig = plt.figure(figurename, figsize=(15, 11))
    fig.suptitle(figurename + ' Haussdorf distance from ground truth for neuron %d' % neuron,
                 fontsize=14, fontweight='bold')
    bbox = [200,850,1200,1720]
    V2size = lambda x : np.abs(x) * 2

    # Map axons for Bakkum's method, high threshold
    ax1 = plt.subplot(331)
    plot_image_axon_delay_voltage(ax1, path+'axon', axon_Bakkum[5], delay_Bakkum, Vmin_Bakkum, x, y, transform=V2size)
    cross_hair(ax1, x_AIS, y_AIS, color='red')
    mea_axes(ax1, bbox=bbox)
    ax1.set_title('Method I')
    plt.text(0.1, 0.9, r'$\mathsf{V_n>5\sigma_{V}}$', ha='left', va='center', transform=ax1.transAxes)

    # Map axons for Bullmann's method, grid spacing ~ 20um
    ax2 = plt.subplot(332)
    ax2h = plot_image_axon_delay_voltage(ax2, path+'axon', axon_Bullmann[1], delay_Bullmann[1], Vmin_Bullmann[1], x_Bullmann[1], y_Bullmann[1], transform=V2size)
    cross_hair(ax2, x_AIS, y_AIS, color='red')
    mea_axes(ax2, bbox=bbox)
    ax2.set_title('Method II')
    plt.text(0.1, 0.9, r'$\mathsf{r\approx20\mu m}$', ha='left', va='center', transform=ax2.transAxes)

    # Ground truth
    ax3 = plt.subplot(333)
    ImageIterator(path).plot()
    cross_hair(ax2, x_AIS, y_AIS, color='red')
    mea_axes(ax3, bbox=bbox)
    ax3.set_title('Groundtruth')

    # Map axons for Bakkum's method, low threshold
    ax4 = plt.subplot(334)
    plot_image_axon_delay_voltage(ax4, path+'axon', axon_Bakkum[3], delay_Bakkum, Vmin_Bakkum, x, y, transform=V2size)
    cross_hair(ax4, x_AIS, y_AIS, color='red')
    mea_axes(ax4, bbox=bbox)
    plt.text(0.1, 0.9, r'$\mathsf{V_n>3\sigma_{V}}$', ha='left', va='center', transform=ax4.transAxes)

    # Map axons for Bullmann's method, grid spacing ~ 40um
    ax5 = plt.subplot(335)
    plot_image_axon_delay_voltage(ax5, path+'axon', axon_Bullmann[2], delay_Bullmann[2], Vmin_Bullmann[2], x_Bullmann[2], y_Bullmann[2], transform=V2size)
    cross_hair(ax5, x_AIS, y_AIS, color='red')
    mea_axes(ax5, bbox=bbox)
    plt.text(0.1, 0.9, r'$\mathsf{r\approx40\mu m}$', ha='left', va='center', transform=ax5.transAxes)

    # Colorbar for A, B, D, E
    ax6 = plt.subplot(336)
    for V in [1, 3, 10, 30, 100]:
        plt.scatter([],[],s=V2size(V), color='white', edgecolor='black', label='%d' % V)
    leg = plt.legend(loc=2, scatterpoints=1, frameon=False, title = r'$\mathsf{V_n\ [\mu V]}\ \ \ \ \ $')
    leg.get_title().set_fontsize(14)
    plt.axis('off')
    adjust_position(ax6,xshrink=0.05,yshrink=0.02,xshift=-0.04)
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes("right", size="10%", pad=0.05)

    norm = mpl.colors.Normalize(vmin=0, vmax=2)
    cbar = mpl.colorbar.ColorbarBase(cax,
                                     cmap=plt.cm.summer,
                                     norm=norm,
                                     orientation='vertical')
    cbar.set_label(r'$\mathsf{\tau_{axon}\ [ms]}$', fontsize=14)

    # Reading groundtruth xg, yg from the axon label file(s)
    xg, yg = ImageIterator(path + 'axon').truth()

    # Bakkum's method: Haussdorf distance vs. threshold
    ax7 = plt.subplot(337)
    xx = list()
    yy = list()
    for pnr in axon_Bakkum.keys():
        xx.append(pnr)
        yy.append(compare_with_groundtruth(x[axon_Bakkum[pnr]], y[axon_Bakkum[pnr]], xg, yg))
    plt.plot (xx,yy,'o--')
    plt.xlabel(r'$\mathsf{V_{thr}\ [\sigma_{V}]}$')
    plt.ylabel(r'$\mathsf{H\ [\mu m]}$')
    plt.ylim((0,400))
    plt.xlim((6,1))
    adjust_position(ax7, xshrink=0.01, yshrink=0.01)
    without_spines_and_ticks(ax7)

    # Bullmann's method: Haussdorf distance vs. spacing of electrodes in the hexagonal grid
    ax8 = plt.subplot(338)
    xx = list()
    yy = list()
    for period in axon_Bullmann.keys():
        xx.append(period * 20)
        yy.append(compare_with_groundtruth(x_Bullmann[period][axon_Bullmann[period]], y_Bullmann[period][axon_Bullmann[period]], xg, yg))
    plt.plot(xx, yy, 'o--')
    plt.xlabel(r'$\mathsf{r\ [\mu m]}$')
    plt.ylabel(r'$\mathsf{H\ [\mu m]}$')
    plt.ylim((0,400))
    plt.xlim((0,80))
    adjust_position(ax8, xshrink=0.01, yshrink=0.01)
    without_spines_and_ticks(ax8)

    label_subplot(ax1, 'A', xoffset=-0.04, yoffset=-0.015)
    label_subplot(ax2, 'B', xoffset=-0.04, yoffset=-0.015)
    label_subplot(ax3, 'C', xoffset=-0.04, yoffset=-0.015)
    label_subplot(ax4, 'D', xoffset=-0.04, yoffset=-0.015)
    label_subplot(ax5, 'E', xoffset=-0.04, yoffset=-0.015)
    label_subplot(ax7, 'F', xoffset=-0.05, yoffset=-0.015)
    label_subplot(ax8, 'G', xoffset=-0.05, yoffset=-0.015)

    plt.show()


def compare_with_groundtruth(x, y, xg, yg):
    C1 = np.array([x, y]).T
    C2 = np.array([xg, yg]).T
    distance = distanceBetweenCurves(C1, C2)
    return distance


def plot_image_axon_delay_voltage(ax, path, axon, delay, V, x, y, transform=np.abs):
    ImageIterator(path).plot()
    radius = transform(V) if transform else V
    s = ax.scatter(x[axon], y[axon], s=radius[axon], c=delay[axon], marker='o', cmap=plt.cm.summer, vmin=0, vmax=2, alpha=0.8)
    return s


def neighbors_from_electrode_positions(x, y, neighborhood_radius = 20):
    # only a subset of electrodes used TODO: generalize and merge electrode_neighborhoods

    pos_as_array = np.asarray(zip(x, y))
    distances = squareform(pdist(pos_as_array, metric='euclidean'))
    neighbors = distances < neighborhood_radius
    return neighbors


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))