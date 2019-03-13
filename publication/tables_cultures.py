import logging
import os
import pickle
import yaml
from matplotlib import pyplot as plt

from hana.misc import unique_neurons
from hana.plotting import plot_neuron_points, mea_axes, plot_neuron_id, plot_network
from hana.recording import load_positions
from hana.segmentation import load_compartments, neuron_position_from_trigger_electrode

from publication.data import Experiment, FIGURE_CULTURES
from simulation import Simulation
from publication.plotting import show_or_savefig, \
    plot_loglog_fit, without_spines_and_ticks, adjust_position, plot_correlation, plot_synapse_delays
from publication.figure_effective import DataFrame_from_Dicts
from collections import Counter

logging.basicConfig(level=logging.DEBUG)


def make_table(figurename, figpath=None):

    for culture in FIGURE_CULTURES:

        # report = Experiment(culture).report()
        # metadata = Experiment(culture).metadata()

        logging.info('Load neurons for culture %d' % culture)
        trigger_el, AIS_el, axon_delay, dendrite_peak = Experiment(culture).compartments()
        pos = load_positions(mea='hidens')
        # electrode_area = average_electrode_area(pos)
        trigger_pos = neuron_position_from_trigger_electrode(pos, trigger_el)
        AIS_pos = neuron_position_from_trigger_electrode(pos, AIS_el)

        # counting number of neurons per electrode
        el_counts = Counter(map(int, trigger_el.values()))

        # counting number of electrodes with n neurons each
        cluster_count = Counter(el_counts.values())
        total_neurons = len(AIS_el)

        total_units = len (el_counts)
        single_units = cluster_count[1]
        multi_units = total_units- single_units
        no_units = 62 - total_units

        print ( no_units, single_units, multi_units, total_neurons )
        print (cluster_count)



if __name__ == "__main__":
    make_table(os.path.basename(__file__))


