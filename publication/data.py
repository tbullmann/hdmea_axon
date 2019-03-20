from publication.experiment import Experiment

FIGURE_CULTURE = 1
FIGURE_NEURON = 5

# culture 1 with spikesorting, approx 46 neurons
FIGURE_NEURONS = [1, 2, 3, 4, 5, 10, 15, 16, 17, 19, 21, 23, 25, 33, 34, 35, 36, 42, 44, 47, 48, 52, 55, 57, 59, 61,
                  65, 66, 73, 76, 80, 82, 84, 89, 97, 100, 103, 104, 105, 107, 109, 114, 116, 121, 124, 127]

FIGURE_CULTURES = [1, 2]

GROUND_TRUTH_CULTURE = 2
GROUND_TRUTH_NEURON = 1544

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
