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




# Testing code

def testing_load_traces(path):


    # Get traves

    V, t, x, y, trigger, neuron = load_traces(path+'.h5')
    if trigger<0:  # may added to load_traces with trigger>-1 as condition
        trigger = find_AIS(V)

    t *= 1000  # convert to ms
    time = 1  #ms
    print t

    # Position of the AIS
    index_AIS = find_AIS(V)
    x_AIS = x[index_AIS]
    y_AIS = y[index_AIS]

    V = np.min(V, axis=1)

    print np.shape(V)

    # Plotting
    fig = plt.figure('Figure X', figsize=(13, 7))
    fig.suptitle('Figure X. Ground truth', fontsize=14, fontweight='bold')

    # Map voltage
    ax4 = plt.subplot(111)
    show_tiles(path)
    # ax4_h1 = ax4.scatter(x, y, c=V, s=15, marker='o', edgecolor='None', cmap='seismic')
    ax4_h1 = ax4.scatter(x, y, s=-V, c='blue', alpha=0.2)

    # voltage_color_bar(ax4_h1, vmin=-40, vmax=40, vstep=10, label=r'$V$ [$\mu$V]')
    # # cross_hair(ax4, x_AIS, y_AIS, color='red')
    # plt.axis('equal')
    # ax4.invert_yaxis()
    set_axis_hidens(ax4)

    plt.show()



def show_tiles(path):
    textfilename = os.path.join(path,'tilespecs.txt')
    data = pd.read_table(textfilename)
    for index, row in data.iterrows():
        filename, left, right, bottom, top = row[['filename','xstart','xend','ystart','yend']]
        tile = plt.imread(os.path.join(path,filename))
        print filename
        plt.imshow(np.sqrt(tile), cmap='gray_r', extent=[left, right, top, bottom])





testing_load_traces('data/groundtruth/neuron1544')
testing_load_traces('data/groundtruth/neuron1536')


