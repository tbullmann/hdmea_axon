import figure_axon
import figure_groundtruth
import figure_comparison
import figure_dendrite
import figure_structural_and_functional
import figure_structur_vs_function_and_synapses
import figure_multiple_networks
import figure_all_neurons
import figure_more_structure_vs_function
import figure_polychronous_groups

if __name__ == "__main__":

    figpath = 'figures'

    figure_axon.make_figure('Figure1', figpath=figpath)
    # figure_groundtruth.make_figure('Figure2', figpath=figpath)
    figure_comparison.make_figure('Figure3', figpath=figpath)
    figure_dendrite.make_figure('Figure4', figpath=figpath)
    figure_structural_and_functional.make_figure('Figure5', figpath=figpath)
    figure_structur_vs_function_and_synapses.make_figure('Figure6', figpath=figpath)
    figure_polychronous_groups.make_figure('Figure7', figpath=figpath)

    figure_multiple_networks.make_figure('Supplementary Figure 1', figpath=figpath)
    figure_all_neurons.make_figure('Supplementary Figure 2', figpath=figpath)
    figure_more_structure_vs_function.make_figure('Supplementary Figure 3', figpath=figpath)
