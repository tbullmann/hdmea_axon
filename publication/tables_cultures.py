import logging
import os

from hana.recording import load_positions
from hana.segmentation import neuron_position_from_trigger_electrode

from data import FIGURE_CULTURES
from experiment import Experiment
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


