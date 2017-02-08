import numpy as np
from itertools import product
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
from hana.recording import load_traces, load_positions
from hana.plotting import set_axis_hidens, plot_neuron_points, plot_neuron_id
from hana.segmentation import load_compartments, load_neurites, neuron_position_from_trigger_electrode
from publication.plotting import FIGURE_NEURON_FILE, FIGURE_NEURON_FILE_FORMAT, FIGURE_NEURONS, FIGURE_ARBORS_FILE, \
    label_subplot, voltage_color_bar, cross_hair, adjust_position


def figure_01():
    fig = plt.figure('Figure 1', figsize=(13, 11))
    fig.suptitle('Figure 1. Experiment Outline', fontsize=14,
                 fontweight='bold')

    ax1 = plt.subplot2grid((2, 5), (0, 0), rowspan=2)

    img = plt.imread('data/outline.png')
    plt.imshow(img)
    plt.axis('off')

    # ---- Plot AIS signals for each neuron showing no overlap ---
    trigger, AIS_index, axon_delay, dendrite_peak = load_compartments(FIGURE_ARBORS_FILE)
    pos = load_positions(mea='hidens')
    neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)

    ax2 = plt.subplot2grid((2, 5), (0, 1), rowspan=1, colspan=2)
    plot_activity_map(FIGURE_NEURON_FILE_FORMAT, FIGURE_NEURONS)
    plot_neuron_points(ax2, trigger, neuron_pos)
    plot_neuron_id(ax2, trigger, neuron_pos)
    ax2.text(250, 250, r'isocontours for -50$\mu$V at 0ms', size=12, bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    set_axis_hidens(ax2)
    adjust_position(ax2, xshrink=0.01, yshrink=0.01)

    # ---- Plot traveling signals for one neuron ---

    # Load example data for one neuron
    V, t, x, y, trigger, neuron = load_traces(FIGURE_NEURON_FILE)
    print(trigger)
    t *= 1000  # convert to ms

    ax3 = plt.subplot2grid((2, 5), (0, 3), rowspan=1, colspan=2)
    # plot_sequential_isopotentials(t, V, x, y, t_max=2.2)
    plot_sequential_isopotentials(t, V, x, y, t_max=2.2)
    ax3.legend(loc=4,frameon=False,prop={'size':12})
    ax3.text(250, 250, r'isocontours for -5$\mu$V', size=12, bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    cross_hair(ax3, x[int(trigger)], y[int(trigger)], color='black')
    set_axis_hidens(ax3)
    adjust_position(ax3, xshrink=0.01, yshrink=0.01)

    ax4 = plt.subplot2grid((2, 5), (1, 1), rowspan=1, colspan=4)
    plot_array_of_stills (t, V, x, y)
    plt.axis('off')
    ax4.set_aspect('equal')
    ax4.invert_yaxis()
    adjust_position(ax4, xshrink=-0.03, yshrink=-0.03)

    label_subplot(ax1, 'A', xoffset=-0.005, yoffset=-0.025)
    label_subplot(ax2, 'B', xoffset=-0.035, yoffset=-0.015)
    label_subplot(ax3, 'C', xoffset=-0.035, yoffset=-0.015)
    label_subplot(ax4, 'D', xoffset=0.005, yoffset=-0.035)

    plt.show()


def plot_activity_map(template, neurons):
    mpl.rcParams['contour.negative_linestyle'] = 'solid'

    new_x_range = np.arange(175.5 , 1908.9, 5)
    new_y_range = np.arange(98.123001, 2096.123, 5)
    new_x, new_y = np.meshgrid(new_x_range, new_y_range, indexing='ij')
    for neuron in neurons:
        # Load  data
        V, t, x, y, trigger, _ = load_traces(template % (neuron))
        frame = V[:, 81]
        print V.shape
        Z = griddata(zip(x, y), frame, (new_x, new_y), method='cubic')
        # CS1 = plt.contourf(new_x, new_y, Z, colors='gray', levels=(-600, -5),zorder=2*i)
        CS2 = plt.contour(new_x, new_y, Z, colors='red', levels=(-50,))
        plt.plot(x[int(trigger)],y[int(trigger)],'k.')


def plot_sequential_isopotentials(t, V, x, y, t_min=0, t_max=None, isopotentials=(-5,)):
    if not t_max:
        t_max = max(t)
    mpl.rcParams['contour.negative_linestyle'] = 'solid'
    new_x_range = np.arange(min(x), max(x), 5)
    new_y_range = np.arange(min(y), max(y), 5)
    new_x, new_y = np.meshgrid(new_x_range, new_y_range, indexing='ij')

    indices = list(*np.where(np.logical_and(t >= t_min, t <= t_max)))
    color = iter(cm.rainbow(np.linspace(0, 1, len(indices))))
    for i in indices:
        c = next(color)
        if i % 10 == 0:
            plt.vlines(-10, -10, -10, colors=c, label='%1.1fms' % t[i])  # proxy for contour line legend
        frame = V[:, i]
        Z = griddata(zip(x, y), frame, (new_x, new_y), method='cubic')
        # CS1 = plt.contourf(new_x, new_y, Z, colors='gray', levels=(-600, -5),zorder=2*i)
        CS2 = plt.contour(new_x, new_y, Z, colors=(c,), levels=isopotentials)


def plot_array_of_stills(t, V, x, y, t_min=-0.4, period=0.2, layout=(5, 3), margin=200):
    start_index = np.where(t==t_min)[0]
    delta_index = int(period/(t[1]-t[0]))
    delta_x = (1908.9 - 175.5 + margin)
    delta_y = (2096.123 - 98.123001 + margin)
    V_all, x_all, y_all = [],[],[]
    for index, (y_index, x_index) in enumerate(product(range(layout[1]),range(layout[0]))):
        frame_index = start_index + index * delta_index
        frame = V[:, frame_index]
        x_all.extend(x + x_index * delta_x)
        y_all.extend(y + y_index * delta_y)
        V_all.extend(frame)
        plt.text(x_index * delta_x + 200, y_index * delta_y + 250,
             '%1.1fms' % t[frame_index], bbox=dict(facecolor='white', pad=1, edgecolor='none'))

    h1 = plt.scatter(x_all, y_all, c=V_all, s=10, marker='o', edgecolor='None', cmap='seismic')
    plt.xlim([min(x_all) - margin, max(x_all) + margin])
    plt.ylim([min(y_all) - margin, max(y_all) + margin])
    voltage_color_bar(h1, label = r'$V$ [$\mu$V]')



figure_01()



