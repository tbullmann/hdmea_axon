import logging
import os

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import squareform, pdist

from hana.grid import HidensTransformation
from hana.plotting import mea_axes
from hana.recording import load_traces
from hana.segmentation import segment_axon_verbose, find_AIS
from publication.comparison import segment_axon_Bakkum, ImageIterator, distanceBetweenCurves
from publication.data import Experiment, FIGURE_CULTURE, FIGURE_NEURON, GROUND_TRUTH_CULTURE, GROUND_TRUTH_NEURON
from publication.plotting import show_or_savefig, cross_hair, adjust_position, without_spines_and_ticks

logging.basicConfig(level=logging.DEBUG)


def make_figure(figurename, figpath=None):

    # Get traces
    V, t, x, y, trigger, neuron = Experiment(GROUND_TRUTH_CULTURE).traces(GROUND_TRUTH_NEURON)
    if trigger<0:  # may added to load_traces with trigger>-1 as condition
        trigger = find_AIS(V)

    t *= 1000  # convert to ms

    # AIS coordinates
    index_AIS = find_AIS(V)
    x_AIS = x[index_AIS]
    y_AIS = y[index_AIS]

    # Segmentation according Bakkum
    axon_Bakkum = dict()
    for pnr in (2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5):
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
    fig = plt.figure(figurename, figsize=(13, 9))
    if not figpath:
        fig.suptitle(figurename + ' Haussdorf distance from ground truth for neuron %d' % neuron,
                 fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.05)

    bbox = None
    if neuron==1544:
        bbox = [200,850,1200,1720]
    V2size = lambda x : np.abs(x) * 2

    # Map axons for Bakkum's method, high threshold
    ax1 = plt.subplot(331)
    plot_image_axon_delay_voltage(ax1,
                                  Experiment(GROUND_TRUTH_CULTURE).images(GROUND_TRUTH_NEURON, type='axon'),
                                  axon_Bakkum[5], delay_Bakkum, Vmin_Bakkum, x, y, transform=V2size)
    cross_hair(ax1, x_AIS, y_AIS, color='red')
    mea_axes(ax1, bbox=bbox, barposition='inside')
    ax1.set_title('Method I')
    plt.text(0.1, 0.9, r'$\mathsf{V_n>5\sigma_{V}}$', ha='left', va='center', transform=ax1.transAxes)
    plt.title('a', loc='left', fontsize=18)

    # Map axons for Bullmann's method, grid spacing ~ 20um
    ax2 = plt.subplot(332)
    ax2h = plot_image_axon_delay_voltage(ax2,
                                         Experiment(GROUND_TRUTH_CULTURE).images(GROUND_TRUTH_NEURON, type='axon'),
                                         axon_Bullmann[1], delay_Bullmann[1], Vmin_Bullmann[1], x_Bullmann[1],
                                         y_Bullmann[1], transform=V2size)
    cross_hair(ax2, x_AIS, y_AIS, color='red')
    mea_axes(ax2, bbox=bbox, barposition='inside')
    ax2.set_title('Method II')
    plt.text(0.1, 0.9, r'$\mathsf{r\approx18\mu m}$', ha='left', va='center', transform=ax2.transAxes)
    plt.title('b', loc='left', fontsize=18)

    # Ground truth
    ax3 = plt.subplot(333)
    Experiment(GROUND_TRUTH_CULTURE).images(GROUND_TRUTH_NEURON).plot()
    cross_hair(ax2, x_AIS, y_AIS, color='red')
    mea_axes(ax3, bbox=bbox, barposition='inside')
    ax3.set_title('Groundtruth')
    plt.title('c', loc='left', fontsize=18)

    # Map axons for Bakkum's method, low threshold
    ax4 = plt.subplot(334)
    plot_image_axon_delay_voltage(ax4,
                                  Experiment(GROUND_TRUTH_CULTURE).images(GROUND_TRUTH_NEURON, type='axon'),
                                  axon_Bakkum[3], delay_Bakkum, Vmin_Bakkum, x, y, transform=V2size)
    cross_hair(ax4, x_AIS, y_AIS, color='red')
    mea_axes(ax4, bbox=bbox, barposition='inside')
    plt.text(0.1, 0.9, r'$\mathsf{V_n>3\sigma_{V}}$', ha='left', va='center', transform=ax4.transAxes)
    plt.title('d', loc='left', fontsize=18)

    # Map axons for Bullmann's method, grid spacing ~ 40um
    ax5 = plt.subplot(335)
    plot_image_axon_delay_voltage(ax5,
                                  Experiment(GROUND_TRUTH_CULTURE).images(GROUND_TRUTH_NEURON, type='axon'),
                                  axon_Bullmann[2], delay_Bullmann[2], Vmin_Bullmann[2], x_Bullmann[2],
                                  y_Bullmann[2], transform=V2size)
    cross_hair(ax5, x_AIS, y_AIS, color='red')
    mea_axes(ax5, bbox=bbox, barposition='inside')
    plt.text(0.1, 0.9, r'$\mathsf{r\approx36\mu m}$', ha='left', va='center', transform=ax5.transAxes)
    plt.title('e', loc='left', fontsize=18)

    # Colorbar and Size legend for A, B, D, E
    ax6 = plt.subplot(336)
    for V in [1, 3, 10, 30, 100]:
        plt.scatter([],[],s=V2size(V), color='gray', edgecolor='none', label='%d' % V)
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
    xg, yg = Experiment(GROUND_TRUTH_CULTURE).images(GROUND_TRUTH_NEURON, type='axon').truth()

    # Bakkum's method: Haussdorf distance vs. threshold
    ax7 = plt.subplot(337)
    xx = list()
    yy = list()
    for pnr in np.sort(axon_Bakkum.keys()):
        xx.append(pnr)
        yy.append(compare_with_groundtruth(x[axon_Bakkum[pnr]], y[axon_Bakkum[pnr]], xg, yg))
    plt.plot (xx,yy,'ko-')
    plt.xlabel(r'$\mathsf{V_{thr}\ [\sigma_{V}]}$')
    plt.ylabel(r'$\mathsf{H\ [\mu m]}$')
    plt.ylim((0,400))
    plt.xlim((6,1))
    adjust_position(ax7, xshrink=0.01, yshrink=0.01)
    without_spines_and_ticks(ax7)
    plt.title('f', loc='left', fontsize=18)

    # Bullmann's method: Haussdorf distance vs. spacing of electrodes in the hexagonal grid
    ax8 = plt.subplot(338)
    xx = list()
    yy = list()
    for period in axon_Bullmann.keys():
        xx.append(period * 18)
        yy.append(compare_with_groundtruth(x_Bullmann[period][axon_Bullmann[period]], y_Bullmann[period][axon_Bullmann[period]], xg, yg))
    plt.plot(xx, yy, 'ko--')
    plt.xlabel(r'$\mathsf{r\ [\mu m]}$')
    plt.ylabel(r'$\mathsf{H\ [\mu m]}$')
    plt.ylim((0,400))
    plt.xlim((0,80))
    adjust_position(ax8, xshrink=0.01, yshrink=0.01)
    without_spines_and_ticks(ax8)
    plt.title('g', loc='left', fontsize=18)

    ax9 = plt.subplot(339)
    img = plt.imread('larger_neighborhoods.png')
    plt.imshow(img)
    plt.axis('off')
    plt.title('h', loc='left', fontsize=18)

    show_or_savefig(figpath, figurename)


def compare_with_groundtruth(x, y, xg, yg):
    C1 = np.array([x, y]).T
    C2 = np.array([xg, yg]).T
    distance = distanceBetweenCurves(C1, C2)
    return distance


def plot_image_axon_delay_voltage(ax, images, axon, delay, V, x, y, transform=np.abs, alpha=1):
    images.plot(alpha=alpha)
    radius = transform(V) if transform else V
    s = ax.scatter(x[axon], y[axon], s=radius[axon], c=delay[axon],
                   marker='o', edgecolor='none',
                   cmap=plt.cm.summer, vmin=0, vmax=2, alpha=0.8)
    return s


def neighbors_from_electrode_positions(x, y, neighborhood_radius = 20):
    # only a subset of electrodes used TODO: generalize and merge electrode_neighborhoods

    pos_as_array = np.asarray(zip(x, y))
    distances = squareform(pdist(pos_as_array, metric='euclidean'))
    neighbors = distances < neighborhood_radius
    return neighbors


def test(neuron=1536, method=2):

    path ='data/neuron%d' % neuron

    # Get traces
    V, t, x, y, trigger, neuron = load_traces(path+'.h5')
    if trigger<0:  # may added to load_traces with trigger>-1 as condition
        trigger = find_AIS(V)
    t *= 1000  # convert to ms

    print ('Time %1.2f .. %1.2f ms ' % (min(t),max(t)))

    # AIS coordinates
    index_AIS = find_AIS(V)
    x_AIS = x[index_AIS]
    y_AIS = y[index_AIS]

    # Negative peak
    Vmin = np.min(V, axis=1)

    # Segmentation
    if method==1:
        delay, _, _, _, axon = segment_axon_Bakkum(V, t, pnr_threshold=5)
    else:
        neighbors = neighbors_from_electrode_positions(x, y)
        delay, _, std_delay, _, thr, _, _, _, axon = segment_axon_verbose(t, V, neighbors)

    print ('Axonal delay %1.2f .. %1.2f ms ' % ( min(delay[axon]),max(delay[axon])) )

    ax = plt.subplot(111)
    V2size = lambda x : np.abs(x) * 2
    axh = plot_image_axon_delay_voltage(ax, path + 'axon', axon, delay, Vmin, x, y, transform=V2size, alpha=0.5)
    cross_hair(ax, x_AIS, y_AIS, color='red')
    mea_axes(ax)

    # Compare with ground truth
    xg, yg = ImageIterator(path + 'axon').truth()
    H = compare_with_groundtruth(x, y, xg, yg)
    plt.title('Neuron %d, Method %d: H=%1.3f um' % (neuron, method, H) )

    plt.show()


def test_subset():
    V, t, x, y, trigger, neuron = Experiment(FIGURE_CULTURE).traces(FIGURE_NEURON)

    hidens = HidensTransformation(x, y)
    xs, ys, Vs = hidens.subset(V)

    ax = plt.subplot(111)
    plt.plot(x, y, 'o', markerfacecolor='None', markeredgecolor='black', label='all')
    plt.plot(xs, ys, 'ko', label='subset')
    mea_axes(ax)
    plt.legend(numpoints=1)

    plt.show()


def test_grid():
    V, t, x, y, trigger, neuron = Experiment(FIGURE_CULTURE).traces(FIGURE_NEURON)

    hidens = HidensTransformation(x, y)

    i, j = hidens.xy2ij(x,y)
    xb, yb = hidens.ij2xy(i,j)
    x0, y0 = hidens.ij2xy(0,0)

    ax1 = plt.subplot(121)
    plt.plot(x, y, 'bx', label='original')
    plt.plot(xb, yb, 'k+', label='backtransformed')
    plt.plot(x0, y0, 'go', label='hexagonal origin (backtransformed)')
    plt.title ('cartesian coordinates x,y')
    mea_axes(ax1)
    ax1.set_ylim((-500,2200))
    plt.legend(numpoints=1)

    ax2 = plt.subplot(122)
    plt.plot(i, j, 'ko', label='transformed')
    plt.plot(0, 0, 'go', label='hexagonal origin')
    plt.title('hexagonal grid index i, j')
    ax2.set_xlim((-1,np.amax(i)))
    ax2.set_ylim((-1, np.amax(j)))
    plt.legend(numpoints=1)

    plt.show()


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))
    # test()
    # test_grid()
    # test_subset()
