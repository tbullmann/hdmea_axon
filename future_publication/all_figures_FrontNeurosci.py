import figure_outline
import figure_spatial_spread
import figure_axon
import figure_groundtruth
import figure_comparison
import figure_dendrite
import figure_structural
import figure_functional
import figure_structur_vs_function
import figure_synapse
import figure_all_neurons
import figure_more_structure_vs_function

if __name__ == "__main__":
    figpath = 'temp/figures/FrontNeurosci'

    figure_all_neurons.make_figure('Supplementary Figure 1', figpath=figpath)

    figure_outline.make_figure('Figure 1', figpath=figpath)
    figure_spatial_spread.make_figure('Figure 2', figpath=figpath)
    figure_axon.make_figure('Figure 3', figpath=figpath)
    figure_groundtruth.make_figure('Figure 4', figpath=figpath)
    figure_comparison.make_figure('Figure 5', figpath=figpath)
    figure_dendrite.make_figure('Figure 6', figpath=figpath)
    figure_structural.make_figure('Figure 7', figpath=figpath)
    figure_functional.make_figure('Figure 8', figpath=figpath)
    figure_structur_vs_function.make_figure('Figure 9', figpath=figpath)
    figure_synapse.make_figure('Figure 10', figpath=figpath)

    figure_more_structure_vs_function.make_figure('Supplementary Figure 2', figpath=figpath)
