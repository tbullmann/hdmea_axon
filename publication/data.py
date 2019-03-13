from publication.experiment import Experiment

FIGURE_CULTURE = 1
FIGURE_NEURON = 5  # old neuron 5 or other neurons 5, 10, 11, 20, 25, 2, 31, 41; culture 1
FIGURE_NEURONS = [10, 11, 13, 2, 20, 21, 22, 23, 25, 27, 29, 3, 31, 35, 36, 37, 4, 41, 49, 5, 50, 51, 57, 59]  # culture 1
FIGURE_CULTURES = [1, 8]
GROUND_TRUTH_CULTURE = 8
GROUND_TRUTH_NEURON = 1544
# FIGURE_CONNECTED_NEURON = 84 # others might be 74, 124(!),...
# FIGURE_NOT_FUNCTIONAL_CONNECTED_NEURON = 19  # 17, 109,107,121,...
# FIGURE_NOT_STRUCTURAL_CONNECTED_NEURON = 124  # old neuron 59 ( 21 or 36 )
# FIGURE_THRESHOLD_OVERLAP_AREA = 300.  # um2/electrode


def extract_multiple_networks():
    """
    For testing only.
    """

    for culture in FIGURE_CULTURES:
        data = Experiment(culture)
        data.metadata()
        data.neurites()


if __name__ == "__main__":

    extract_multiple_networks()
