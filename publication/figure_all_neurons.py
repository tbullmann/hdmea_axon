from hana.recording import load_positions
from hana.plotting import mea_axes
from data import FIGURE_CULTURE, FIGURE_NEURONS
from experiment import Experiment
from plotting import show_or_savefig, cross_hair, adjust_position, make_axes_locatable

import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)


def make_figure(figurename, figpath=None, Culture=FIGURE_CULTURE, with_background_image=True):

    # make directory
    subfolderpath = os.path.join(figpath, 'neurons')
    if not os.path.exists(subfolderpath):
        os.makedirs(subfolderpath)

    # Load electrode coordinates and calculate neighborhood
    pos = load_positions(mea='hidens')
    x = pos.x
    y = pos.y

    all_triggers, all_AIS, all_axonal_delays, all_dendritic_return_currents = Experiment(Culture).compartments()

    for neuron in FIGURE_NEURONS:

        # Plot figure
        fig = plt.figure(figurename, figsize=(8, 7))
        # for example neuron5.png use instead:
        # fig = plt.figure(figurename)
        # fig.set_size_inches([5.5, 5])

        ax = plt.subplot(111)

        # Load and plot background images
        if with_background_image:
            fullfilename = os.path.join(Experiment(Culture).data_directory, "neuron%d.png" % neuron)
            raw_image = plt.imread(fullfilename)
            plt.imshow(raw_image,
                cmap='gray',
                alpha=0.5,
                extent=[165.5, 1918.9, 2106.123, 88.123001])

        # Map axons for Bullmann's method
        delay = all_axonal_delays[neuron]
        index_AIS = all_AIS[neuron]
        axon = np.isfinite(delay)
        V, _, _, _, _, _ = Experiment(Culture).traces(neuron)
        # max_axon_delay = np.around(np.nanmax(delay), decimals=1)
        max_axon_delay = 2.5 # same scaling for all neurons

        V2size = lambda x: np.sqrt(np.abs(x)) * 5
        radius = V2size(V)
        ax.scatter(x[axon], y[axon], s=radius[axon], c=delay[axon],
                   marker='o', edgecolor='none',
                   cmap=plt.cm.hsv, vmin=0, vmax=max_axon_delay, alpha=1)

        plt.title('Culture %d, Neuron %d, Delay map' % (Culture, neuron))

        if with_background_image:
            # cross_hair(ax, x[index_AIS], y[index_AIS], color='red')
            mea_axes(ax)
        else:
            ax.plot(x[index_AIS], y[index_AIS], marker='x', markersize=20, color='blue')
            mea_axes(ax, style='axes')
            ax.spines['bottom'].set_color('blue')
            ax.spines['top'].set_color('blue')
            ax.spines['right'].set_color('blue')
            ax.spines['left'].set_color('blue')
            ax.tick_params(axis='x', colors='blue')
            ax.tick_params(axis='y', colors='blue')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        # for example neuron5.png use instead:
        # cax = divider.append_axes("right", size="5%", pad=0.05)

        norm = mpl.colors.Normalize(vmin=0, vmax=max_axon_delay)
        cbar = mpl.colorbar.ColorbarBase(cax,
                                         cmap=plt.cm.hsv,
                                         norm=norm,
                                         orientation='vertical')
        cbar.set_label(r'$\mathsf{\tau_{axon}\ [ms]}$', fontsize=14)


        longfigurename = "neuron%0d" % neuron
        show_or_savefig(subfolderpath, longfigurename)
        # for example neuron5.png use instead:
        # show_or_savefig(subfolderpath, longfigurename, dpi = 72)

        print ("Plotted neuron %d" % neuron)

        plt.close()



if __name__ == "__main__":

    #make_figure(os.path.basename(__file__))
    make_figure('spikesorted_footprints', figpath='figures\culture5', Culture=5)

