import logging
import os
import pickle
import yaml

from hana.function import timeseries_to_surrogates, all_timelag_standardscore, all_peaks
from hana.polychronous import filter, shuffle_network
from hana.recording import load_timeseries, partial_timeseries
from hana.segmentation import extract_and_save_compartments, load_neurites
from hana.structure import all_overlaps
from publication.figure_polychronous_groups import extract_pcgs
from publication.plotting import FIGURE_CULTURES, correlate_two_dicts_verbose


def extract_multiple_networks():

    for culture in FIGURE_CULTURES:

        sub_dir = 'culture%d' % culture

        # Maybe create output directory; hdf5 filename for all neurites
        results_directory = os.path.join('temp', sub_dir)
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        # hdf5 filename template for spike triggered average data for each neuron
        data_directory = os.path.join('data2', sub_dir)
        neuron_file_template = os.path.join(data_directory, 'neuron%d.h5')

        # Reading hidens and recording date
        metadata = yaml.load(open(os.path.join('data2', sub_dir, 'metadata.yaml'), 'r'))
        logging.info ('Culture%d on hidens%d recorded on %d' % (culture, metadata['hidens'], metadata['recording']))

        # Maybe extract neurites
        neurites_filename = os.path.join(results_directory, 'all_neurites.h5')
        if not os.path.isfile(neurites_filename):
            extract_and_save_compartments(neuron_file_template, neurites_filename)

        # Maybe extract structural and functional network
        events_filename = os.path.join(data_directory, 'events.h5')
        two_networks_pickle_name = os.path.join(results_directory, 'two_networks.p')
        if not os.path.isfile(two_networks_pickle_name):
            # structural networks
            axon_delay, dendrite_peak = load_neurites(neurites_filename)
            structural_strength, _, structural_delay = all_overlaps(axon_delay, dendrite_peak, thr_peak=0, thr_ratio=0,
                                                                    thr_overlap=1)
            # functional networks: only use forward direction
            neurons_with_axons = axon_delay.keys()
            logging.info('Neurons with axon: {}'.format(neurons_with_axons))
            timeseries = load_timeseries(events_filename, neurons=neurons_with_axons)
            logging.info('Surrogate time series')
            timeseries_surrogates = timeseries_to_surrogates(timeseries, n=20, factor=2)
            logging.info('Compute standard score for histograms')
            timelags, std_score_dict, timeseries_hist_dict = all_timelag_standardscore(timeseries, timeseries_surrogates)
            functional_strength, functional_delay, _, _ = all_peaks(timelags, std_score_dict, thr=1, direction='forward')
            # pickle structural and functional network
            pickle.dump((structural_strength, structural_delay, functional_strength, functional_delay),
                        open(two_networks_pickle_name, 'wb'))
            logging.info('Saved structural and functional network')

        # Maybe extract admissible delays (spike timings for synaptic connections only)
        admissible_delays_pickle_name = os.path.join(results_directory, 'admissible_delays.p')
        if not os.path.isfile(admissible_delays_pickle_name):
            _, structural_delay, _, functional_delay = pickle.load(open(two_networks_pickle_name, 'rb'))
            axonal_delays, spike_timings, pairs = correlate_two_dicts_verbose(structural_delay, functional_delay)
            putative_delays = {pair: timing for delay, timing, pair in zip(axonal_delays, spike_timings, pairs)
                               if timing - delay > 1}  # if synapse_delay = spike_timing - axonal_delay > 1ms
            pickle.dump(putative_delays, open(admissible_delays_pickle_name, 'wb'))

        # Maybe extract partial timeseries TODO More transparent way of using only part of the data!
        partial_timeseries_pickle_name = os.path.join(results_directory, 'partial_timeseries.p')
        if not os.path.isfile(partial_timeseries_pickle_name):
            timeseries = load_timeseries(events_filename)
            for neuron in timeseries:
                print len(timeseries[neuron])
            timeseries = partial_timeseries(timeseries)  # using only first 10% of recording
            for neuron in timeseries:
                print len(timeseries[neuron])
            pickle.dump(timeseries, open(partial_timeseries_pickle_name, 'wb'))

        # Maybe extract connected events
        connected_events_pickle_name = os.path.join(results_directory, 'connected_events.p')
        if not os.path.isfile(connected_events_pickle_name):
            putative_delays = pickle.load(open(admissible_delays_pickle_name, 'rb'))
            timeseries = pickle.load(open(partial_timeseries_pickle_name, 'rb'))

            connected_events = filter(timeseries, putative_delays, additional_synaptic_delay=0,
                                      synaptic_jitter=0.0005)
            pickle.dump(connected_events, open(connected_events_pickle_name, 'wb'))

        # Maybe extract connected events for original time series on surrogate networks = surrogate 1 and 2
        connected_events_surrogate_1_pickle_name = os.path.join(results_directory, 'connected_events_surrogate_1.p')
        if not os.path.isfile(connected_events_surrogate_1_pickle_name):
            putative_delays = pickle.load(open(admissible_delays_pickle_name, 'rb'))
            timeseries = pickle.load(open(partial_timeseries_pickle_name, 'rb'))
            surrogate_delays = shuffle_network(putative_delays, method='shuffle in-nodes')
            connected_surrogate_events = filter(timeseries, surrogate_delays, additional_synaptic_delay=0,
                                                synaptic_jitter=0.0005)
            pickle.dump(connected_surrogate_events, open(connected_events_surrogate_1_pickle_name, 'wb'))
        connected_events_surrogate_2_pickle_name = os.path.join(results_directory, 'connected_events_surrogate_2.p')
        if not os.path.isfile(connected_events_surrogate_2_pickle_name):
            putative_delays = pickle.load(open(admissible_delays_pickle_name, 'rb'))
            timeseries = pickle.load(open(partial_timeseries_pickle_name, 'rb'))
            surrogate_delays = shuffle_network(putative_delays, method='shuffle values')
            connected_surrogate_events = filter(timeseries, surrogate_delays, additional_synaptic_delay=0,
                                                synaptic_jitter=0.0005)
            pickle.dump(connected_surrogate_events, open(connected_events_surrogate_2_pickle_name, 'wb'))

        # original surrogate time series on original network = surrogate 3
        partial_surrogate_timeseries_pickle_name = os.path.join(results_directory,'partial_surrogate_timeseries.p')
        if not os.path.isfile(partial_surrogate_timeseries_pickle_name):
            timeseries = pickle.load(open(partial_timeseries_pickle_name, 'rb'))
            # remove (partial) time series without spikes
            timeseries = {neuron: timeseries[neuron] for neuron in timeseries if len(timeseries[neuron]) > 0}
            surrogate_timeseries = timeseries_to_surrogates(timeseries, n=1, factor=2)
            # keeping only the first of several surrogate times series for each neuron
            surrogate_timeseries = {neuron: timeseries[0] for neuron, timeseries in
                                    surrogate_timeseries.items()}
            pickle.dump(surrogate_timeseries, open(partial_surrogate_timeseries_pickle_name, 'wb'))
        connected_events_surrogate_3_pickle_name = os.path.join(results_directory, 'connected_events_surrogate_3.p')
        if not os.path.isfile(connected_events_surrogate_3_pickle_name):
            putative_delays = pickle.load(open(admissible_delays_pickle_name, 'rb'))
            surrogate_timeseries = pickle.load(open(partial_surrogate_timeseries_pickle_name, 'rb'))
            connected_surrogate_events = filter(surrogate_timeseries, putative_delays, additional_synaptic_delay=0,
                                                synaptic_jitter=0.0005)
            pickle.dump(connected_surrogate_events, open(connected_events_surrogate_3_pickle_name, 'wb'))

        # Maybe extract size distribution for PCGs
        PCG_sizes_pickle_name = os.path.join(results_directory, 'pcg_sizes.p')
        if not os.path.isfile(PCG_sizes_pickle_name):
            _, pcgs_size = extract_pcgs(connected_events_pickle_name)
            _, pcgs1_size = extract_pcgs(connected_events_surrogate_1_pickle_name)
            _, pcgs2_size = extract_pcgs(connected_events_surrogate_2_pickle_name)
            _, pcgs3_size = extract_pcgs(connected_events_surrogate_3_pickle_name)
            pickle.dump((pcgs_size, pcgs1_size, pcgs2_size, pcgs3_size), open(PCG_sizes_pickle_name, 'wb'))