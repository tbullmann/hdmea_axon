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
from publication.figure_synapses import DataFrame_from_Dicts

logging.basicConfig(level=logging.DEBUG)


def make_figure(figurename, figpath=None):

    for culture in FIGURE_CULTURES:

        report = Experiment(culture).report()
        metadata = Experiment(culture).metadata()

        logging.info('Load neuron positions')
        trigger, _, axon_delay, dendrite_peak = Experiment(culture).compartments()
        pos = load_positions(mea='hidens')
        # electrode_area = average_electrode_area(pos)
        neuron_pos = neuron_position_from_trigger_electrode(pos, trigger)

        logging.info('Load tructural and functional network')
        _, structural_delay, _, functional_delay, _, synaptic_delay = Experiment(culture).networks()
        neuron_dict = unique_neurons(structural_delay)
        # Getting and subsetting the data

        logging.info('Load PCG size distribution')
        pcgs_size, pcgs1_size, pcgs2_size, pcgs3_size = Experiment(culture).polychronous_group_sizes_with_surrogates()

        logging.info('Making figure')
        longfigurename = figurename + '-%d' % culture
        fig = plt.figure(longfigurename, figsize=(21, 14))
        if not figpath:
            fig.suptitle(figurename + '-%d  Culture %d: HDMEA hidens%d recorded at %d' % (culture, culture, metadata['hidens'], metadata['recording']), fontsize=14, fontweight='bold')
        plt.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.05)

        logging.info('Plot structural connectiviy')
        ax1 = plt.subplot(231)
        plot_network(ax1, structural_delay, neuron_pos, color='gray')
        plot_neuron_points(ax1, neuron_dict, neuron_pos)
        plot_neuron_id(ax1, neuron_dict, neuron_pos)
        # ax1.text(200, 250, r'$\mathsf{\rho=%3d\ \mu m^2}$' % thr_overlap_area, fontsize=18)
        mea_axes(ax1)
        plt.title('a     structural connectivity', loc='left', fontsize=18)

        logging.info('Plot functional connectiviy')
        ax2 = plt.subplot(232)
        plot_network(ax2, functional_delay, neuron_pos, color='gray')
        plot_neuron_points(ax2, neuron_dict, neuron_pos)
        plot_neuron_id(ax2, neuron_dict, neuron_pos)
        # ax2.text(200, 250, r'$\mathsf{\zeta=%d}$' % thr_z_score, fontsize=18)
        mea_axes(ax2)
        plt.title('b     functional connectivity', loc='left', fontsize=18)

        logging.info('Plot Synaptic delay graph')
        ax5 = plt.subplot(233)
        plot_network(ax5, synaptic_delay, neuron_pos, color='green')
        plot_neuron_points(ax5, unique_neurons(structural_delay), neuron_pos)
        plot_neuron_id(ax5, neuron_dict, neuron_pos)
        mea_axes(ax5)
        adjust_position(ax5, yshift=-0.01)
        plt.title('c     synaptic connectivity', loc='left', fontsize=18)


        _, _, _, sim_functional_delay, _, sim_synaptic_delay = Simulation(culture).networks()

        logging.info('Plot simulated functional connectiviy')
        ax2 = plt.subplot(235)
        plot_network(ax2, sim_functional_delay, neuron_pos, color='gray')
        plot_neuron_points(ax2, neuron_dict, neuron_pos)
        plot_neuron_id(ax2, neuron_dict, neuron_pos)
        # ax2.text(200, 250, r'$\mathsf{\zeta=%d}$' % thr_z_score, fontsize=18)
        mea_axes(ax2)
        plt.title('d     simulated functional connectivity', loc='left', fontsize=18)

        logging.info('Plot simulated synaptic delay graph')
        ax5 = plt.subplot(236)
        plot_network(ax5, sim_synaptic_delay, neuron_pos, color='green')
        plot_neuron_points(ax5, unique_neurons(structural_delay), neuron_pos)
        plot_neuron_id(ax5, neuron_dict, neuron_pos)
        mea_axes(ax5)
        adjust_position(ax5, yshift=-0.01)
        plt.title('e     simulated synaptic connectivity', loc='left', fontsize=18)

        report.close()
        show_or_savefig(figpath, longfigurename)


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))


