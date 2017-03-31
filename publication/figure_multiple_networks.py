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
from publication.plotting import show_or_savefig, \
    plot_loglog_fit, without_spines_and_ticks, adjust_position, plot_correlation, plot_synapse_delays, \
    DataFrame_from_Dicts

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
        structural_strength, structural_delay, functional_strength, functional_delay \
            = Experiment(culture).networks()
        neuron_dict = unique_neurons(structural_delay)
        # Getting and subsetting the data
        data = DataFrame_from_Dicts(functional_delay, functional_strength, structural_delay, structural_strength)
        delayed = data[data.delayed]
        simultaneous = data[data.simultaneous]

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
        plt.title('a     structural connectivity graph', loc='left', fontsize=18)

        logging.info('Plot functional connectiviy')
        ax2 = plt.subplot(232)
        plot_network(ax2, functional_delay, neuron_pos, color='gray')
        plot_neuron_points(ax2, neuron_dict, neuron_pos)
        plot_neuron_id(ax2, neuron_dict, neuron_pos)
        # ax2.text(200, 250, r'$\mathsf{\zeta=%d}$' % thr_z_score, fontsize=18)
        mea_axes(ax2)
        plt.title('b     functional connectivity graph', loc='left', fontsize=18)

        logging.info('Plot synapse delays')
        ax4 = plt.subplot(235)
        plot_synapse_delays(ax4, data, ylim=(-2, 7), report=report)
        plt.title('e     synaptic delays', loc='left', fontsize=18)

        logging.info('Plot correlation between strength')
        ax3 = plt.subplot(234)
        axScatter1 = plot_correlation(ax3, data, x='structural_strength', y='functional_strength',
                                      xscale='log', yscale='log', report=report)
        axScatter1.set_xlabel(r'$\mathsf{|A \cap D|\ [\mu m^2}$]', fontsize=14)
        axScatter1.set_ylabel(r'$\mathsf{z_{max}}$', fontsize=14)
        plt.title('d     correlation of overlap and z-score', loc='left', fontsize=18)

        logging.info('Plot Synaptic delay graph')
        ax5 = plt.subplot(233)
        plot_network(ax5, zip(simultaneous.pre, simultaneous.post), neuron_pos, color='red')
        plot_network(ax5, zip(delayed.pre, delayed.post), neuron_pos, color='green')
        plot_neuron_points(ax5, unique_neurons(structural_delay), neuron_pos)
        plot_neuron_id(ax5, neuron_dict, neuron_pos)
        # Legend by proxy
        ax5.hlines(0, 0, 0, linestyle='-', color='red', label='<1ms')
        ax5.hlines(0, 0, 0, linestyle='-', color='green', label='>1ms')
        plt.legend(frameon=True, fontsize=12)
        mea_axes(ax5)
        adjust_position(ax5, yshift=-0.01)
        plt.title('c     synaptic delay graph', loc='left', fontsize=18)

        logging.info('Plot PCG size distribution')
        ax6 = plt.subplot(236)
        plot_loglog_fit(ax6, pcgs_size, fit=False)
        plot_loglog_fit(ax6, pcgs1_size, datamarker='r-', datalabel='surrogate network 1', fit=False)
        plot_loglog_fit(ax6, pcgs2_size, datamarker='g-', datalabel='surrogate network 2', fit=False)
        plot_loglog_fit(ax6, pcgs3_size, datamarker='b-', datalabel='surrogate timeseries', fit=False)
        ax6.set_ylabel('size [number of spikes / polychronous group]')
        without_spines_and_ticks(ax6)
        ax6.set_ylim((1, 1000))
        ax6.set_xlim((1, 13000))
        plt.title('f     size distribution of spike patterns', loc='left', fontsize=18)

        report.close()
        show_or_savefig(figpath, longfigurename)


if __name__ == "__main__":
    make_figure(os.path.basename(__file__))


