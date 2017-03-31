import logging
import os
import pickle
import yaml

from hana.function import timeseries_to_surrogates, all_timelag_standardscore, all_peaks
from hana.polychronous import filter, shuffle_network, extract_pcgs
from hana.recording import load_timeseries, partial_timeseries, load_traces
from hana.segmentation import extract_and_save_compartments, load_neurites
from hana.structure import all_overlaps

from publication.plotting import FIGURE_CULTURES, correlate_two_dicts_verbose


class Experiment():

    def __init__(self, culture, data_base_dir='data2', temp_base_dir='temp'):
        self.culture = culture
        self.data_base_dir = data_base_dir
        self.temp_base_dir = temp_base_dir
        self.sub_dir = 'culture%d' % culture
        self.data_directory = os.path.join(self.data_base_dir, self.sub_dir)
        self.neuron_file_template = os.path.join(self.data_directory, 'neuron%d.h5')

        self.results_directory = os.path.join(self.temp_base_dir, self.sub_dir)
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

    def metadata(self):
        # Reading hidens and recording date
        metadata = yaml.load(open(os.path.join(self.data_base_dir, self.sub_dir, 'metadata.yaml'), 'r'))
        metadata['culture'] = self.culture
        logging.info ('Culture%d on hidens%d recorded on %d' % (metadata['culture'], metadata['hidens'], metadata['recording']))
        return metadata

    def load_traces(self, neuron):
        return load_traces(self.neuron_file_template % neuron)

    def neurites(self):
        # Maybe extract neurites
        neurites_filename = os.path.join(self.results_directory, 'all_neurites.h5')
        if not os.path.isfile(neurites_filename):
            extract_and_save_compartments(self.neuron_file_template, neurites_filename)
        axon_delay, dendrite_peak = load_neurites(neurites_filename)
        return  axon_delay, dendrite_peak

    def timeseries(self):
        events_filename = os.path.join(self.data_directory, 'events.h5')
        axon_delay, dendrite_peak = self.neurites()
        neurons_with_axons = axon_delay.keys()
        logging.info('Neurons with axon: {}'.format(neurons_with_axons))
        return load_timeseries(events_filename, neurons=neurons_with_axons)

    def networks(self):
        # Maybe extract structural and functional network
        events_filename = os.path.join(self.data_directory, 'events.h5')
        two_networks_pickle_name = os.path.join(self.results_directory, 'two_networks.p')
        if not os.path.isfile(two_networks_pickle_name):
            # structural networks
            axon_delay, dendrite_peak = self.neurites()
            structural_strength, _, structural_delay = all_overlaps(axon_delay, dendrite_peak, thr_peak=0, thr_ratio=0,
                                                                    thr_overlap=1)
            # functional networks: only use forward direction
            timeseries = self.timeseries()
            logging.info('Surrogate time series')
            timeseries_surrogates = timeseries_to_surrogates(timeseries, n=20, factor=2)
            logging.info('Compute standard score for histograms')
            timelags, std_score_dict, timeseries_hist_dict = all_timelag_standardscore(timeseries, timeseries_surrogates)
            functional_strength, functional_delay, _, _ = all_peaks(timelags, std_score_dict, thr=1, direction='forward')
            # pickle structural and functional network
            pickle.dump((structural_strength, structural_delay, functional_strength, functional_delay),
                        open(two_networks_pickle_name, 'wb'))
            logging.info('Saved structural and functional network')
        else:
            structural_strength, structural_delay, functional_strength, functional_delay = \
                pickle.load(open(two_networks_pickle_name, 'rb'))
        return structural_strength, structural_delay, functional_strength, functional_delay

    def putative_delays(self):
        # Maybe extract admissible delays (spike timings for synaptic connections only)
        admissible_delays_pickle_name = os.path.join(self.results_directory, 'admissible_delays.p')
        if not os.path.isfile(admissible_delays_pickle_name):
            _, structural_delay, _, functional_delay = self.networks()
            axonal_delays, spike_timings, pairs = correlate_two_dicts_verbose(structural_delay, functional_delay)
            putative_delays = {pair: timing for delay, timing, pair in zip(axonal_delays, spike_timings, pairs)
                               if timing - delay > 1}  # if synapse_delay = spike_timing - axonal_delay > 1ms
            pickle.dump(putative_delays, open(admissible_delays_pickle_name, 'wb'))
        else:
            putative_delays = pickle.load(open(admissible_delays_pickle_name, 'rb'))
        return putative_delays

    def partial_timeseries(self, interval=0.1):
        # Maybe extract partial timeseries TODO More transparent way of using only first 10% of the data!
        partial_timeseries_pickle_name = os.path.join(self.results_directory, 'partial_timeseries.p')
        if not os.path.isfile(partial_timeseries_pickle_name):
            timeseries = self.timeseries()
            for neuron in timeseries:
                print len(timeseries[neuron])
            timeseries = partial_timeseries(timeseries, interval=interval)
            for neuron in timeseries:
                print len(timeseries[neuron])
            pickle.dump(timeseries, open(partial_timeseries_pickle_name, 'wb'))
        else:
            timeseries = pickle.load(open(partial_timeseries_pickle_name, 'rb'))
        return timeseries

    def connected_events(self, surrogate=None):
        # Maybe extract connected events
        suffix = '_surrogate_%d' % surrogate if surrogate else ''
        connected_events_pickle_name = os.path.join(self.results_directory, 'connected_events'+suffix+'.p')
        if not os.path.isfile(connected_events_pickle_name):
            putative_delays = self.putative_delays()
            if surrogate==1: putative_delays = shuffle_network(putative_delays, method='shuffle in-nodes')
            if surrogate==2: putative_delays = shuffle_network(putative_delays, method='shuffle values')
            timeseries = self.partial_timeseries()
            if surrogate==3:
                timeseries = {neuron: timeseries[neuron] for neuron in timeseries if len(timeseries[neuron]) > 0}
                surrogate_timeseries = timeseries_to_surrogates(timeseries, n=1, factor=2)
                # keeping only the first of several surrogate times series for each neuron
                timeseries = {neuron: timeseries[0] for neuron, timeseries in surrogate_timeseries.items()}
            connected_events = filter(timeseries, putative_delays, additional_synaptic_delay=0,
                                      synaptic_jitter=0.0005)
            pickle.dump(connected_events, open(connected_events_pickle_name, 'wb'))
        else:
            connected_events = pickle.load(open(connected_events_pickle_name, 'rb'))
        return connected_events

    def polychronous_groups(self):
        # Extract PCGs and size distribution
        PCG_pickle_name = os.path.join(self.results_directory, 'pcgs_and_size.p')
        if not os.path.isfile(PCG_pickle_name):
            pcgs, pcgs_size = extract_pcgs(self.connected_events())
            pickle.dump((pcgs, pcgs_size), open(PCG_pickle_name, 'wb'))
        else:
            pcgs, pcgs_size = pickle.load(open(PCG_pickle_name, 'rb'))
        return pcgs, pcgs_size

    def polychronous_group_sizes_with_surrogates (self):
        # Maybe extract size distribution for PCGs including surrogates
        PCG_sizes_pickle_name = os.path.join(self.results_directory, 'pcg_sizes.p')
        if not os.path.isfile(PCG_sizes_pickle_name):
            pcgs, pcgs_size = self.polychronous_groups()
            _, pcgs1_size = extract_pcgs(self.connected_events(surrogate=1))
            _, pcgs2_size = extract_pcgs(self.connected_events(surrogate=2))
            _, pcgs3_size = extract_pcgs(self.connected_events(surrogate=3))
            pickle.dump((pcgs_size, pcgs1_size, pcgs2_size, pcgs3_size), open(PCG_sizes_pickle_name, 'wb'))
        else:
            pcgs_size, pcgs1_size, pcgs2_size, pcgs3_size = pickle.load(open(PCG_sizes_pickle_name, 'rb'))
        return pcgs_size, pcgs1_size, pcgs2_size, pcgs3_size


def extract_multiple_networks():

    for culture in FIGURE_CULTURES:
        data = Experiment(culture)
        data.metadata()
        data.neurites()
        data.timeseries()
        data.networks()
        data.putative_delays()
        data.polychronous_group_sizes_with_surrogates()


if __name__ == "__main__":

    extract_multiple_networks()
