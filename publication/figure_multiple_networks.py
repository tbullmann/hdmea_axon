from hana.recording import load_positions
from hana.plotting import mea_axes
from hana.segmentation import extract_and_save_compartments, load_compartments, load_neurites
from publication.plotting import show_or_savefig, cross_hair, adjust_position, FIGURE_NEURONS

import os
import numpy as np
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)




# Extract multiple networks

def test_neurite_extraction():
    # from os import path, listdir
    # import re
    #
    # path_to_directories = 'data2'
    # dir_regex = 'hidens(\d+)'
    # for sub_dir in listdir(path_to_directories):
    #     if re.compile(dir_regex).match(sub_dir):
    #         print path.join(path_to_directories,sub_dir,'neuron%d.h5')

    neuron_file_template = 'data2/hidens1666/neuron%d.h5'
    neurites_filename = 'temp/hidens1666_all_neurites.h5'

    extract_and_save_compartments(neuron_file_template, neurites_filename)

    trigger, AIS, delay, positive_peak = load_neurites(neurites_filename)
    print('NEW: neuron indices as keys')
    print(trigger.keys())


def test_bug_in_half_pak_width():
    """
    INFO:root:Load file data2/hidens1666_thr6/neuron32.h5 with variables [u'V', u'n', u'neuron', u't', u'trigger', u'x', u'y']
    /Users/tbullmann/anaconda/envs/hdmea/lib/python2.7/site-packages/scipy/interpolate/_fitpack_impl.py:731: RuntimeWarning: The number of zeros exceeds mest
      warnings.warn(RuntimeWarning("The number of zeros exceeds mest"))
    Traceback (most recent call last):
      File "/Users/tbullmann/PycharmProjects/hdmea/publication/figure_multiple_networks.py", line 30, in <module>
        test_neurite_extraction()
      File "/Users/tbullmann/PycharmProjects/hdmea/publication/figure_multiple_networks.py", line 21, in test_neurite_extraction
        extract_and_save_compartments(neuron_file_template, neurites_filename)
      File "/Users/tbullmann/PycharmProjects/hdmea/hana/segmentation.py", line 107, in extract_and_save_compartments
        all_triggers, all_AIS, all_axonal_delays, all_dendritic_return_currents =  extract_all_compartments(neurons, template)
      File "/Users/tbullmann/PycharmProjects/hdmea/hana/segmentation.py", line 42, in extract_all_compartments
        number_dendrite_electrodes = extract_compartments(t, V, neighbors)
      File "/Users/tbullmann/PycharmProjects/hdmea/hana/segmentation.py", line 71, in extract_compartments
        dendrite_return_current = segment_dendrite(t, V, neighbors)
      File "/Users/tbullmann/PycharmProjects/hdmea/hana/segmentation.py", line 186, in segment_dendrite
        return_current_delay, dendrite = segment_dendrite_verbose(t, V, neighbors)
      File "/Users/tbullmann/PycharmProjects/hdmea/hana/segmentation.py", line 169, in segment_dendrite_verbose
        min_delay, max_delay = half_peak_domain(t, V_AIS)
      File "/Users/tbullmann/PycharmProjects/hdmea/hana/recording.py", line 42, in half_peak_domain
        domain = list(roots[range(index_roots-1,index_roots+1)])
    IndexError: index 10 is out of bounds for axis 1 with size 10
    """
    from hana.segmentation import load_traces, extract_compartments, electrode_neighborhoods
    neighbors = electrode_neighborhoods(mea='hidens')
    V, t, x, y, trigger, _ = load_traces('data2/hidens1666/neuron32.h5')
    t *= 1000  # convert to ms

    axon, dendrite, axonal_delay, dendrite_return_current, index_AIS, number_axon_electrodes, \
    number_dendrite_electrodes = extract_compartments(t, V, neighbors)


if __name__ == "__main__":
    test_neurite_extraction()
    test_bug_in_half_pak_width()

