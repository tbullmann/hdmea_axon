import logging
import os
import pickle
import yaml
from matplotlib import pyplot as plt

from hana.misc import unique_neurons
from hana.recording import load_timeseries, partial_timeseries, load_positions, average_electrode_area
from hana.segmentation import extract_and_save_compartments, load_compartments, load_neurites, neuron_position_from_trigger_electrode
from hana.structure import all_overlaps
from hana.function import timeseries_to_surrogates, all_timelag_standardscore, all_peaks
from hana.polychronous import filter, shuffle_network
from hana.plotting import plot_neuron_points, mea_axes, plot_neuron_id, plot_network

from publication.plotting import FIGURE_CULTURES, correlate_two_dicts_verbose, show_or_savefig, plot_loglog_fit, \
    without_spines_and_ticks, adjust_position
from publication.figure_polychronous_groups import extract_pcgs

from figure_structur_vs_function import plot_correlation
from figure_synapse import plot_synapse_delays

logging.basicConfig(level=logging.DEBUG)


# Extract multiple networks

def extract_multiple_networks(cultures=FIGURE_CULTURES):

    print FIGURE_CULTURES

    for culture in cultures:

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
            print metadata

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


def make_figure(figurename, figpath=None,
                thr_overlap_area=3000.,  # um2/electrode
                thr_z_score=10):

    for culture in FIGURE_CULTURES:

        sub_dir = 'culture%d' % culture
        results_directory = os.path.join('temp', sub_dir)
        data_directory = os.path.join('data2', sub_dir)

        # Reading hidens and recording date
        metadata = yaml.load(open(os.path.join('data2', sub_dir, 'metadata.yaml'), 'r'))

        # Load neuron positions
        neurites_filename = os.path.join(results_directory, 'all_neurites.h5')
        trigger, _, axon_delay, dendrite_peak = load_compartments(neurites_filename)
        pos = load_positions(mea='hidens')
        # electrode_area = average_electrode_area(pos)
        neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)

        # Load tructural and functional network
        two_networks_pickle_name = os.path.join(results_directory, 'two_networks.p')
        structural_strength, structural_delay, functional_strength, functional_delay \
            = pickle.load(open(two_networks_pickle_name, 'rb'))
        neuron_dict = unique_neurons(structural_delay)

        # Calculate synaptic delays


        # Load PCG size distribution
        PCG_sizes_pickle_name = os.path.join(results_directory, 'pcg_sizes.p')
        pcgs_size, pcgs1_size, pcgs2_size, pcgs3_size = pickle.load(open(PCG_sizes_pickle_name, 'rb'))

        # Making figure
        fig = plt.figure(figurename + '-hidens%d' % metadata['hidens'] + ' on %d' % metadata['recording'], figsize=(21, 14))
        if not figpath:
            fig.suptitle(figurename + '  Summary for hidens%d' % metadata['hidens'] + ' on %d' % metadata['recording'], fontsize=14, fontweight='bold')
        plt.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.05)

        # Plot structural connectiviy
        ax1 = plt.subplot(231)
        plot_network(ax1, structural_delay, neuron_pos)
        plot_neuron_points(ax1, neuron_dict, neuron_pos)
        plot_neuron_id(ax1, neuron_dict, neuron_pos)
        # ax1.text(200, 250, r'$\mathsf{\rho=%3d\ \mu m^2}$' % thr_overlap_area, fontsize=18)
        mea_axes(ax1)
        plt.title('a     structural connectivity graph', loc='left', fontsize=18)

        # Plot functional connectiviy
        ax2 = plt.subplot(232)
        plot_network(ax2, functional_delay, neuron_pos)
        plot_neuron_points(ax2, neuron_dict, neuron_pos)
        plot_neuron_id(ax2, neuron_dict, neuron_pos)
        # ax2.text(200, 250, r'$\mathsf{\zeta=%d}$' % thr_z_score, fontsize=18)
        mea_axes(ax2)
        plt.title('b     functional connectivity graph', loc='left', fontsize=18)

        # Plot correlation between strength
        ax3 = plt.subplot(234)
        axScatter1 = plot_correlation(ax3, structural_strength, functional_strength, xscale='log', yscale='log',
                                      dofit=True)
        axScatter1.set_xlabel(r'$\mathsf{|A \cap D|\ [\mu m^2}$]', fontsize=14)
        axScatter1.set_ylabel(r'$\mathsf{z_{max}}$', fontsize=14)
        plt.title('c     correlation of overlap and z-score', loc='left', fontsize=18)

        # Plot synapse delays
        ax4 = plt.subplot(235)
        delayed_pairs, simultaneous_pairs, synapse_delays = plot_synapse_delays(ax4, structural_delay,
                                                                                functional_delay,
                                                                                functional_strength, ylim=(-2, 7))
        plt.title('c     synaptic delays', loc='left', fontsize=18)

        # Plot Synaptic delay graph
        ax5 = plt.subplot(233)
        plot_network(ax5, simultaneous_pairs, neuron_pos, color='red')
        plot_network(ax5, delayed_pairs, neuron_pos, color='green')
        plot_neuron_points(ax5, unique_neurons(structural_delay), neuron_pos)
        plot_neuron_id(ax2, trigger, neuron_pos)
        # Legend by proxy
        ax5.hlines(0, 0, 0, linestyle='-', color='red', label='<1ms')
        ax5.hlines(0, 0, 0, linestyle='-', color='green', label='>1ms')
        ax5.text(200, 250, r'$\mathsf{\rho=300\mu m^2}$', fontsize=18)
        ax5.text(200, 350, r'$\mathsf{\zeta=1}$', fontsize=18)
        plt.legend(frameon=False)
        mea_axes(ax5)
        adjust_position(ax5, yshift=-0.01)
        plt.title('c     synaptic delay graph', loc='left', fontsize=18)

        # Plot PCG size distribution
        ax6 = plt.subplot(236)
        plot_loglog_fit(ax6, pcgs_size, fit=False)
        plot_loglog_fit(ax6, pcgs1_size, datamarker='r-', datalabel='surrogate network 1', fit=False)
        plot_loglog_fit(ax6, pcgs2_size, datamarker='g-', datalabel='surrogate network 2', fit=False)
        plot_loglog_fit(ax6, pcgs3_size, datamarker='b-', datalabel='surrogate timeseries', fit=False)
        ax6.set_ylabel('size [number of spikes / polychronous group]')
        without_spines_and_ticks(ax6)
        ax6.set_ylim((1, 1000))
        ax6.set_xlim((1, 13000))
        plt.title('d     size distribution of spike patterns', loc='left', fontsize=18)

        show_or_savefig(figpath, figurename)


if __name__ == "__main__":
    extract_multiple_networks()
    make_figure(os.path.basename(__file__))


