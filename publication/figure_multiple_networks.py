import logging
import os
import pickle

from hana.recording import load_timeseries, partial_timeseries
from hana.segmentation import extract_and_save_compartments, load_neurites
from hana.structure import all_overlaps
from hana.function import timeseries_to_surrogates, all_timelag_standardscore, all_peaks
from hana.polychronous import filter, combine, group, plot_pcg_on_network, plot_pcg, shuffle_network, plot_pcgs
from publication.plotting import FIGURE_CULTURES, correlate_two_dicts_verbose
from publication.figure_polychronous_groups import extract_pcgs
logging.basicConfig(level=logging.DEBUG)


# Extract multiple networks

def extract_multiple_networks(cultures=FIGURE_CULTURES):

    print FIGURE_CULTURES

    for culture in cultures:
            sub_dir = 'hidens%d' % culture

            # Maybe create output directory; hdf5 filename for all neurites
            results_directory = os.path.join('temp', sub_dir)
            if not os.path.exists(results_directory):
                os.makedirs(results_directory)

            # hdf5 filename template for spike triggered average data for each neuron
            data_directory = os.path.join('data2', sub_dir)
            neuron_file_template = os.path.join(data_directory, 'neuron%d.h5')

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
                timeseries = {neuron: timeseries[pair] for neuron in timeseries if len(timeseries[neuron]) > 0}
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


def test_bug_in_find_valley():
    """
    INFO:root:Load file data2/hidens2018/neuron1.h5 with variables [u'V', u'n', u'neuron', u't', u'trigger', u'x', u'y']
    Traceback (most recent call last):
      File "/Users/tbullmann/PycharmProjects/hdmea/publication/figure_multiple_networks.py", line 73, in <module>

      File "/Users/tbullmann/PycharmProjects/hdmea/publication/figure_multiple_networks.py", line 32, in test_neurite_extraction
        print('NEW: neuron indices as keys')
      File "/Users/tbullmann/PycharmProjects/hdmea/hana/segmentation.py", line 107, in extract_and_save_compartments
        all_triggers, all_AIS, all_axonal_delays, all_dendritic_return_currents =  extract_all_compartments(neurons, template)
      File "/Users/tbullmann/PycharmProjects/hdmea/hana/segmentation.py", line 42, in extract_all_compartments
        number_dendrite_electrodes = extract_compartments(t, V, neighbors)
      File "/Users/tbullmann/PycharmProjects/hdmea/hana/segmentation.py", line 69, in extract_compartments
        axonal_delay = segment_axon(t, V, neighbors)
      File "/Users/tbullmann/PycharmProjects/hdmea/hana/segmentation.py", line 219, in segment_axon
        _, mean_delay, _, _, _, _, _, _, axon = segment_axon_verbose(t, V, neighbors)
      File "/Users/tbullmann/PycharmProjects/hdmea/hana/segmentation.py", line 204, in segment_axon_verbose
        thr = find_valley(std_delay, expected_std_delay)
      File "/Users/tbullmann/PycharmProjects/hdmea/hana/recording.py", line 82, in find_valley
        hist, bin_edges = np.histogram(std_delay, bins=np.arange(0, expected_std_delay, step=DELAY_EPSILON))
      File "/Users/tbullmann/anaconda/envs/hdmea/lib/python2.7/site-packages/numpy/lib/function_base.py", line 628, in histogram
        sa.searchsorted(bins[-1], 'right')]
    IndexError: index -1 is out of bounds for axis 0 with size 0
    :return:
    """
    from hana.segmentation import load_traces, extract_compartments, electrode_neighborhoods
    neighbors = electrode_neighborhoods(mea='hidens')
    V, t, x, y, trigger, _ = load_traces('data2/hidens2018/neuron1.h5')
    t *= 1000  # convert to ms

    axon, dendrite, axonal_delay, dendrite_return_current, index_AIS, number_axon_electrodes, \
    number_dendrite_electrodes = extract_compartments(t, V, neighbors)


if __name__ == "__main__":
    extract_multiple_networks()
    # test_bug_in_half_pak_width()
    # test_bug_in_find_valley()

