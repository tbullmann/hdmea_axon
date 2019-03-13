import logging
import os
import pickle

import pandas as pd
import yaml

from hana.function import timeseries_to_surrogates, all_timelag_standardscore, all_peaks
from hana.polychronous import filter, shuffle_network, extract_pcgs
from hana.recording import load_timeseries, partial_timeseries, load_traces, electrode_neighborhoods
from hana.segmentation import extract_and_save_compartments, load_neurites, load_compartments
from hana.structure import all_overlaps

from publication.plotting import correlate_two_dicts_verbose
from publication.comparison import ImageIterator, ModelDiscriminatorBakkum, ModelDiscriminatorBullmann

FIGURE_CULTURE = 2
FIGURE_NEURON = 5  # old neuron 5 or other neurons 5, 10, 11, 20, 25, 2, 31, 41; culture 1
FIGURE_NEURONS = [2, 3, 4, 5, 10, 11, 13, 20, 21, 22, 23, 25, 27, 29, 31, 35, 36, 37, 41, 49, 50, 51, 59]  # culture 1
FIGURE_CULTURES = [1, 2, 3, 4, 5]
GROUND_TRUTH_CULTURE = 8
GROUND_TRUTH_NEURON = 1544
FIGURE_CONNECTED_NEURON = 84 # others might be 74, 124(!),...
FIGURE_NOT_FUNCTIONAL_CONNECTED_NEURON = 19  # 17, 109,107,121,...
FIGURE_NOT_STRUCTURAL_CONNECTED_NEURON = 124  # old neuron 59 ( 21 or 36 )
FIGURE_THRESHOLD_OVERLAP_AREA = 300.  # um2/electrode


class Experiment():
    """
    Provides access to original and extracted data for the recording of a culture.
    """

    def __init__(self, culture, data_base_dir='data', temp_base_dir='temp'):
        self.culture = culture
        self.data_base_dir = data_base_dir
        self.temp_base_dir = temp_base_dir
        self.sub_dir = 'culture%d' % culture
        self.data_directory = os.path.join(self.data_base_dir, self.sub_dir)
        self.neuron_file_template = os.path.join(self.data_directory, 'neuron%d.h5')
        self.results_directory = os.path.join(self.temp_base_dir, self.sub_dir)
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)
        self.neurites_filename = os.path.join(self.results_directory, 'all_neurites.h5')

    def metadata(self):
        """
        Reading the original metadata from a text/YAML file containing the description for the recording of a culture.
        :return: dictionary from YAML file
        """
        # Reading hidens and recording date
        metadata = yaml.load(open(os.path.join(self.data_base_dir, self.sub_dir, 'metadata.yaml'), 'r'))
        metadata['culture'] = self.culture
        logging.info ('Culture%d on hidens%d recorded on %d' % (metadata['culture'], metadata['hidens'], metadata['recording']))
        return metadata

    def traces(self, neuron):
        """
        Spike triggered averages for a neuron.
        :param neuron: neuron index
        :return: traces
        """
        return load_traces(self.neuron_file_template % neuron)

    def images(self, neuron, type=''):
        """
        Images with ground truth for a neuron.
        :param neuron: neuron index
        :param type: '': original images
                     'axon': axon tracing
        :return: ImageIterator
        """
        path = os.path.join(self.data_directory, 'neuron%d' % neuron + type)
        return ImageIterator(path)

    def neurites(self):
        """See self.compartments."""
        if not os.path.isfile(self.neurites_filename):
            extract_and_save_compartments(self.neuron_file_template, self.neurites_filename)
        axon_delay, dendrite_peak = load_neurites(self.neurites_filename)
        return axon_delay, dendrite_peak

    def compartments(self):
        """
        Extract compartments.
        :return: Dictionaries indexed by neurons:
            triggers: electrode used for triggering the spike-triggered averages
            AIS: electrode near the (proximal) axon inital segment (AIS)
            delays: axonal delays
            positive_peak: dendritic peak positive voltage (representing the return)
        """
        if not os.path.isfile(self.neurites_filename):
            extract_and_save_compartments(self.neuron_file_template, self.neurites_filename)
        triggers, AIS, delays, positive_peak = load_compartments(self.neurites_filename)
        return triggers, AIS, delays, positive_peak

    def timeseries(self):
        """
        Load time series for all neurons, and return only those with axons.
        :return: time series indexed by neurons
        """
        events_filename = os.path.join(self.data_directory, 'events.h5')
        axon_delay, dendrite_peak = self.neurites()
        neurons_with_axons = axon_delay.keys()
        logging.info('Neurons with axon: {}'.format(neurons_with_axons))
        return load_timeseries(events_filename, neurons=neurons_with_axons)

    def timeseries_surrogates(self):
        """
        Calculate surrogates for original time series.
        :return: dictionary indexed by neurons
        """
        timesseries_surrogates_filename = os.path.join(self.results_directory, 'events_surrogates.p')
        if not os.path.isfile(timesseries_surrogates_filename):
            logging.info('Surrogate time series')
            timeseries_surrogates = timeseries_to_surrogates(self.timeseries())
            pickle.dump(timeseries_surrogates, open(timesseries_surrogates_filename, 'wb'))
        else:
            timeseries_surrogates = pickle.load(open(timesseries_surrogates_filename, 'rb'))
        return timeseries_surrogates

    def standardscores(self):
        """
        Extract standard score for spike timings.
        :return:
        timelags: time lags used for computation of histograms
        std_score_dict: standard scores indexed by pairs of pre and post-synaptic neurons:
        timeseries_hist_dict: histograms (spike counts) indexed by pairs of pre and post-synaptic neurons:
        """
        standardscores_filename = os.path.join(self.results_directory, 'standardscores.p')
        if not os.path.isfile(standardscores_filename):
            logging.info('Compute standard score for histograms')
            timelags, std_score_dict, timeseries_hist_dict = all_timelag_standardscore(self.timeseries(),
                                                                                       self.timeseries_surrogates())
            pickle.dump((timelags, std_score_dict, timeseries_hist_dict), open(standardscores_filename, 'wb'))
        else:
            timelags, std_score_dict, timeseries_hist_dict = pickle.load(open(standardscores_filename, 'rb'))
        return timelags, std_score_dict, timeseries_hist_dict

    def networks(self):
        """
        Extract structural, functional and synaptic network
        :return: dictionaries indexed by pairs of pre and post-synaptic neurons:
            structural_strength: overlap (area) between pre-synaptic axon and post-synaptic dendrite
            structural_delay: delay by axons
            functional_strength: maximal z-score (for functional connectivity)
            functional_delay: spike timing
            synaptic_strength: maximal z-score (for synaptic connectivity)
            synaptic_delay: response delay (spike timing - delay by axons)
        """
        three_networks_pickle_name = os.path.join(self.results_directory, 'three_networks.p')
        if not os.path.isfile(three_networks_pickle_name):
            logging.info('Could not find structural, functional and synaptic network in')
            logging.info(three_networks_pickle_name)
            logging.info('Calculate structural, functional and synaptic network')
            structural_delay, structural_strength = self.structural_network()
            functional_delay, functional_strength = self.functional_network()
            synaptic_delay, synaptic_strength = self.synaptic_network()
            pickle.dump((structural_strength, structural_delay,
                         functional_strength, functional_delay,
                         synaptic_strength, synaptic_delay),
                        open(three_networks_pickle_name, 'wb'))
            logging.info('Saved structural, functional and synaptic network')
        else:
            structural_strength, structural_delay, \
            functional_strength, functional_delay, \
            synaptic_strength, synaptic_delay = \
                pickle.load(open(three_networks_pickle_name, 'rb'))
        return structural_strength, structural_delay, \
               functional_strength, functional_delay,  \
               synaptic_strength, synaptic_delay

    def synaptic_network(self):
        # synaptic networks: use axonal delays from structural connectivity and a minimal synapstic delays of 1.0 ms
        timelags, std_score_dict, timeseries_hist_dict = self.standardscores()
        structural_delay, structural_strength = self.structural_network()
        synaptic_strength, synaptic_delay, _ = all_peaks(timelags, std_score_dict,
                                                         structural_delay_dict=structural_delay,
                                                         minimal_synapse_delay=1.0)
        return synaptic_delay, synaptic_strength

    def functional_network(self):
        # functional networks: only use forward direction
        timelags, std_score_dict, timeseries_hist_dict = self.standardscores()
        functional_strength, functional_delay, _ = all_peaks(timelags, std_score_dict)
        return functional_delay, functional_strength

    def structural_network(self):
        # structural networks
        axon_delay, dendrite_peak = self.neurites()
        structural_strength, _, structural_delay = all_overlaps(axon_delay, dendrite_peak, thr_peak=0, thr_ratio=0,
                                                                thr_overlap=1)
        return structural_delay, structural_strength

    def putative_delays(self):
        """
        Extract admissible delays (spike timings for synaptic connections only).
        :return: delays dictionary indexed by pairs of pre and post-synaptic neurons.
        """
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
        """
        Extract partial timeseries for extraction of polychronous groups. This takes a lot of time, therefore not
        all data is used by default.
        :param interval: part of time series (default: 0.1, first 10% of the data are used)
        :return: time series dictionary indexed by neurons
        """
        partial_timeseries_pickle_name = os.path.join(self.results_directory, 'partial_timeseries.p')
        if not os.path.isfile(partial_timeseries_pickle_name):
            timeseries = self.timeseries()
            timeseries = partial_timeseries(timeseries, interval=interval)
            pickle.dump(timeseries, open(partial_timeseries_pickle_name, 'wb'))
        else:
            timeseries = pickle.load(open(partial_timeseries_pickle_name, 'rb'))
        return timeseries

    def connected_events(self, surrogate=None):
        """
        Extract connected events
        :param surrogate: None: original data
                          1, 2, or 3: surrogates 1, 2, or 3
        :return:
        """
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
        """
        Extract polychronous groups and their sizes for the original data.
        :return: pcgs: polychronous groups
                 pcgs_size: polychronous group sizes
        """
        PCG_pickle_name = os.path.join(self.results_directory, 'pcgs_and_size.p')
        if not os.path.isfile(PCG_pickle_name):
            pcgs, pcgs_size = extract_pcgs(self.connected_events())
            pickle.dump((pcgs, pcgs_size), open(PCG_pickle_name, 'wb'))
        else:
            pcgs, pcgs_size = pickle.load(open(PCG_pickle_name, 'rb'))
        return pcgs, pcgs_size

    def polychronous_group_sizes_with_surrogates (self):
        """
        Extract sizes for the polychronous groups for the original data and surrogates
        :return: pcgs_size: for orginal data
                 pcgs1_size, pcgs2_size, pcgs3_size: for three different surrogates
        """
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

    def comparison_of_discriminators(self):
        """
        Evaluating both models for all neurons, load from csv if already exist.
        :return: pandas data frame with the following columns:
            AUC: area under curve
            FPR, TPR: : false positive rate and true positive rate at the threshold
            n_N, n_P: number of electrodes with signal and without (=background)
            gamma: ratio n_P/(n_N + n_P)
            method: 'I': Bakkum et al.
                    'II': Bullmann et al.
            subject: neuron number
        """
        evaluation_filename = os.path.join(self.results_directory, 'comparison_of_discriminators.csv')

        if os.path.isfile(evaluation_filename):
            data = pd.DataFrame.from_csv(evaluation_filename)

        else:
            Model1 = ModelDiscriminatorBakkum()
            Model2 = ModelDiscriminatorBullmann()

            # Load electrode coordinates
            neighbors = electrode_neighborhoods(mea='hidens')

            evaluations = list()
            for neuron in FIGURE_NEURONS:
                V, t, x, y, trigger, neuron = Experiment(FIGURE_CULTURE).traces(neuron)
                t *= 1000  # convert to ms

                Model1.fit(t, V, pnr_threshold=5)
                Model1.predict()
                Model2.fit(t, V, neighbors)
                Model2.predict()

                evaluations.append(Model1.summary(subject='%d' % neuron, method='I'))
                evaluations.append(Model2.summary(subject='%d' % neuron, method='II'))

            data = pd.DataFrame(evaluations)
            data.to_csv(evaluation_filename)

        return data

    def report(self):
        """
        Evaluating both models for all neurons, load from csv if already exist.
        :return: handle for text/YAML file
        """
        report_filename = os.path.join(self.results_directory, 'report.yaml')
        report = open(report_filename, "w")
        yaml.dump(self.metadata(), report)
        return report


def extract_multiple_networks():
    """
    For testing only.
    """

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
