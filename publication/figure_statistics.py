from hana.recording import load_positions
from hana.recording import electrode_distances

from hana.plotting import mea_axes
from publication.data import FIGURE_CULTURE, FIGURE_NEURONS
from publication.experiment import Experiment
from publication.plotting import show_or_savefig, cross_hair, adjust_position, make_axes_locatable

import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)


def make_figure(figurename, figpath=None, Culture=FIGURE_CULTURE, with_background_image=False):

    # Load electrode coordinates and calculate neighborhood
    pos = load_positions(mea='hidens')
    x = pos.x
    y = pos.y

    distances = electrode_distances()

    all_triggers, all_AIS, all_axonal_delays, all_dendritic_return_currents = Experiment(Culture).compartments()

    all_Neurons = []
    all_Coefficents = []
    all_Count = []
    all_Delay = []
    all_N_gaps = []

    print(FIGURE_NEURONS)

    for neuron in FIGURE_NEURONS:

        # delay ~ distance
        delay = all_axonal_delays[neuron]
        index_AIS = all_AIS[neuron]
        axon = np.isfinite(delay)

        delay_AIS = delay[index_AIS]
        delay = delay[axon]
        min_delay = min(delay) if not np.isfinite(delay_AIS) else delay_AIS

        delay = delay - min_delay
        r = distances[index_AIS, axon]

        # Polynomial fit to delay ~ r
        Coefficents = np.polyfit(r, delay, 3)
        print(Coefficents)

        # Sholl analysis
        R, Count = sholl_analysis(r)
        R, Delay = sholl_analysis(r, delay)

        # Number of large gaps indicative of myelinated parts of axons
        N_gaps = len(axon_gaps(axon, distances))

        # Collect
        all_Neurons.append(neuron)
        all_Coefficents.append(Coefficents)
        all_Count.append(Count)
        all_Delay.append(Delay)
        all_N_gaps.append(N_gaps)

    from collections import Counter
    print(all_N_gaps)
    print(Counter(all_N_gaps))

    # # Colors for neurons
    # cNorm = mpl.colors.Normalize(vmin=1, vmax=max(FIGURE_NEURONS))
    # scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('hsv'))
    # color = scalarMap.to_rgba(neuron)
    # color = 'black'

    fig = plt.figure(figurename, figsize=(9, 5))
    ax1 = plt.subplot(121)
    ax1.plot(R, np.array(all_Count).T, '-' )
    ax1.set_xlim((0,1400))
    ax1.set_ylabel('number of electrodes with axonal signal')
    ax1.set_xlabel(r'distance from AIS in $\mathsf{\mu m}$')
    adjust_position(ax1, xshrink=0.02)
    plt.title('a', loc='left', fontsize=18)


    ax2 = plt.subplot(122)
    ax2.plot(R, np.array(all_Delay).T, '-' )
    ax2.set_xlim((0,1400))
    ax2.set_ylabel(r'axonal delay $\tau_{axon}$ in ms')
    ax2.set_xlabel(r'distance from AIS in $\mathsf{\mu m}$')
    adjust_position(ax2, xshrink=0.02)
    plt.title('b', loc='left', fontsize=18)

    show_or_savefig(figpath, figurename)
    # plt.close()


def axon_gaps(axon, distances, minimum_gap_length = 300):
    """
    The axonal arbor is reconstructed from the minimum spanning tree containing all electrodes with axonal signals
    Most gaps are equal to the shortest distance between electrodes, however some electrodes with axonals are far
    from other electrodes with axonal signals, indicating a "jump", which might indicate saltatory conduction.
    There is a typical length of the myelinated stretches of axons (hence larger than a minimum_gap_length).
    :param axon: electrodes with axonal signals
    :param distances: matrix of inter-electrode distances
    :param minimum_gap_length: threshold for gap length to be reported
    :return:
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    X = distances[axon][:, axon]    # gives a square, which distances[axon, axon] gives not!
    X = csr_matrix(X)
    Tcsr = minimum_spanning_tree(X)
    min_tree_distances = Tcsr.toarray().ravel()
    min_tree_distances = min_tree_distances[min_tree_distances > minimum_gap_length]
    return min_tree_distances


def sholl_analysis(r, value=[], fun=np.mean, min_n=5, shell_thickness=100, shell_count=14):
    """
    Quick implementation of the classic Sholl analysis.
    :param r: radius of sample
    :param value: values of sample, can be None
    :param shell_thickness:
    :param shell_count:
    :return x: mid distance for each shell
    :return y: mean value for each shell, if None is given the number of samples/Shell is counted
    """
    shell_range = range(shell_thickness, shell_thickness * shell_count, shell_thickness)
    shell = np.ceil(r / shell_thickness) * shell_thickness
    n = [sum(shell == i) for i in shell_range]
    n = np.array(n)
    if value == []:
        y = n
    else:
        y = [ fun(value[shell == i]) for i in shell_range ]
        y = np.array(y)
        y[n<min_n] = np.nan
    x = np.array(shell_range) - shell_thickness // 2
    return x, np.array(y)


if __name__ == "__main__":

    #make_figure(os.path.basename(__file__))
    make_figure('statistics', figpath='figures')

