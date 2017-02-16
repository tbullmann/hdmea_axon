import numpy as np
import os
import pickle
from itertools import product
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
from hana.recording import load_traces, load_positions
from hana.plotting import mea_axes, plot_neuron_points, plot_neuron_id
from hana.segmentation import load_compartments, load_neurites, neuron_position_from_trigger_electrode
from publication.plotting import FIGURE_NEURON_FILE, FIGURE_NEURON_FILE_FORMAT, FIGURE_NEURONS, FIGURE_ARBORS_FILE, \
    label_subplot, voltage_color_bar, cross_hair
from future_publication._frangi import frangi

FIGURE_INTERPOLATION_FILE = 'temp/interpolation.p'
FIGURE_RIDGE_FILE = 'temp/ridge.p'


def figure_S03():

    # Load example data for one neuron
    V, t, x, y, trigger, neuron = load_traces(FIGURE_NEURON_FILE)
    t *= 1000  # convert to ms

    if not os.path.isfile(FIGURE_INTERPOLATION_FILE):

        # Negative peak
        V_min = np.min(V[:, 81:-1], axis=1)

        # Interpolate on grid
        new_x_range = np.arange(175.5, 1908.9, 5)
        new_y_range = np.arange(98.123001, 2096.123, 5)
        new_x, new_y = np.meshgrid(new_x_range, new_y_range, indexing='ij')
        new_V = griddata(zip(x, y), V_min, (new_x, new_y), method='cubic')

        # Pickel
        pickle.dump((new_x, new_y, new_V), open(FIGURE_INTERPOLATION_FILE, 'wb'))
    else:
        new_x, new_y, new_V = pickle.load(open(FIGURE_INTERPOLATION_FILE, 'rb'))

    if  os.path.isfile(FIGURE_RIDGE_FILE):

        # Ridge detection by Frangi's filter
        # Zf = frangi(Z, scale_range=(1, 5), scale_step=1, beta1=0.5, beta2=15,
        #            black_ridges=True)
        ridge = frangi(new_V, scale_range=(1, 3), scale_step=1, beta1=2.0, beta2=10,
                       black_ridges=True)

        # Replace not a number by zero
        ridge[np.where(np.isnan(ridge))] = 0

        # Pickel
        pickle.dump(ridge, open(FIGURE_RIDGE_FILE, 'wb'))
    else:
        ridge = pickle.load(open(FIGURE_RIDGE_FILE, 'rb'))



    # Plotting
    mpl.rcParams['contour.negative_linestyle'] = 'solid'
    fig = plt.figure('Figure 1', figsize=(11, 8))
    fig.suptitle('Figure S3. Compartmental model for simulation', fontsize=14,
                 fontweight='bold')

    ax1 = plt.subplot(221)
    h1 = plt.scatter(new_x, new_y, c=new_V, s=10, marker='o', edgecolor='None', cmap='seismic')
    # CS1 = plt.contourf(new_x, new_y, Z, colors='gray', levels=(-600, -5),zorder=2*i)
    # CS2 = plt.contour(new_x, new_y, Z, colors='red', levels=(-5,))
    voltage_color_bar(h1)
    plt.plot(x[int(trigger)],y[int(trigger)],'k.')
    mea_axes(ax1)

    ax2 = plt.subplot(222)
    transformed_ridge = 1-np.sqrt(ridge)
    transformed_ridge[np.where(np.isnan(transformed_ridge))]=0

    # plt.imshow(transformed_ridge.T, cmap='gray')
    h2 = plt.scatter(new_x, new_y, c=transformed_ridge, s=10, marker='o', edgecolor='None', cmap='gray')
    plt.colorbar(h2)
    plt.plot(x[int(trigger)],y[int(trigger)],'k.')
    mea_axes(ax2)

    ax3 = plt.subplot(223)
    h3 = plt.scatter(new_x, new_y, c=transformed_ridge>0.95, s=10, marker='o', edgecolor='None', cmap='gray')
    plt.plot(x[int(trigger)],y[int(trigger)],'k.')
    mea_axes(ax3)


    ax3 = plt.subplot(224)
    plt.hist(transformed_ridge.ravel(),bins=256)
    plt.show()

figure_S03()


