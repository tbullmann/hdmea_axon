import os, pickle

import numpy as np

from hana.matlab import load_neurites
from hana.structure import all_overlaps


# Data preparation

def explore_parameter_space_for_structural_connectivity():
    axon_delay, dendrite_peak = load_neurites ('data/hidens2018at35C_arbors.mat')

    resolution = 3
    alpha = np.float64(range(-5*resolution,5*resolution+1))/resolution
    thresholds_overlap = list((2**alpha)/(2**alpha+1))
    resolution = resolution*2
    thresholds_peak  = list(2**(np.float(exp+1)/resolution)-1 for exp in range(5*resolution+1))

    print 'total', len(thresholds_peak), 'thresholds for peak ', thresholds_peak
    print 'total', len(thresholds_overlap), 'thresholds for overlap = ', thresholds_overlap

    networks = []

    for thr_peak in thresholds_peak:
        for thr_overlap in thresholds_overlap:
            print 'Connections for peak > %1.1f mV and overlap > = %1.2f' % (thr_peak, thr_overlap)
            all_ratio, all_delay = all_overlaps(axon_delay, dendrite_peak, thr_peak=thr_peak, thr_overlap=thr_overlap)
            k = len(all_ratio)
            print 'Network connection k = %d' % k
            networks.append(((thr_peak, thr_overlap),{'overlap ratio': all_ratio, 'delay': all_delay}))

    print 'Finished exploring %d different parameter sets' % len(networks)

    pickle.dump(dict(networks), open('temp/struc_networks_for_thr_peak_thr_overlap_hidens2018.p', 'wb'))

    print 'Saved data'


# Plotting



if not os.path.isfile('temp/struc_networks_for_thr_peak_thr_overlap_hidens2018.p'): explore_parameter_space_for_structural_connectivity()


