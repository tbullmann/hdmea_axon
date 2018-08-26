import figure_functional
import figure_structural
import figure_functional_vs_structural
import figure_synapses
import figure_graphs
import figure_dynamics

if __name__ == "__main__":

    figpath = 'figures'

    figure_functional.make_figure('Figure2', figpath=figpath)
    figure_structural.make_figure('Figure3', figpath=figpath)
    figure_functional_vs_structural.make_figure('Figure4', figpath=figpath)
    figure_synapses.make_figure('Figure5', figpath=figpath)
    figure_graphs.make_figure('Figure6', figpath=figpath)
    figure_dynamics.make_figure('SupplementalFigure1', figpath=figpath)
