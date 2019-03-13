import figure_footprint
import figure_axon
import figure_groundtruth
import figure_comparison
import figure_all_neurons
import figure_dendrite

if __name__ == "__main__":

    figpath = 'figures'

    figure_footprint.make_figure('Figure2', figpath=figpath)
    # figure_axon.make_figure('Figure3', figpath=figpath)
    # figure_groundtruth.make_figure('Figure4', figpath=figpath)
    # figure_comparison.make_figure('Figure5', figpath=figpath)
    # figure_all_neurons.make_figure('Figure6', figpath=figpath) #Should be more an example
    # figure_dendrite.make_figure('Figure7', figpath=figpath)
