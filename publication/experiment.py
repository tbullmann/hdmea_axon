import os

import pandas as pd
import yaml

from hana.recording import electrode_neighborhoods
from comparison import ImageIterator, ModelDiscriminatorBakkum, ModelDiscriminatorBullmann
from hana.experiment import Experiment


class AxonExperiment(Experiment):
    """
    Provides access to original and extracted data for the recording of a culture.
    """

    def images(self, neuron, type=''):
        """
        Images with ground truth for a neuron.
        :param neuron: neuron index
        :param type: '': original images
                     'axon': axon tracing
        :return: ImageIterator
        """
        path = os.path.join(self.data_directory, 'neuron%d' % neuron + type)
        return ImageIterator(path)

    def comparison_of_discriminators(self, compared_neurons):
        """
        Evaluating both models for all neurons, load from csv if already exist.
        :return: pandas data frame with the following columns:
            AUC: area under curve
            FPR, TPR: : false positive rate and true positive rate at the threshold
            n_N, n_P: number of electrodes with signal and without (=background)
            gamma: ratio n_P/(n_N + n_P)
            method: 'I': Bakkum et al.
                    'II': Bullmann et al.
            subject: neuron number
        """
        evaluation_filename = os.path.join(self.results_directory, 'comparison_of_discriminators.csv')

        if os.path.isfile(evaluation_filename):
            data = pd.DataFrame.from_csv(evaluation_filename)

        else:
            Model1 = ModelDiscriminatorBakkum()
            Model2 = ModelDiscriminatorBullmann()

            # Load electrode coordinates
            neighbors = electrode_neighborhoods(mea='hidens')

            evaluations = list()
            for neuron in compared_neurons:
                V, t, x, y, trigger, neuron = AxonExperiment(self.culture).traces(neuron)
                t *= 1000  # convert to ms

                Model1.fit(t, V, pnr_threshold=5)
                Model1.predict()
                Model2.fit(t, V, neighbors)
                Model2.predict()

                evaluations.append(Model1.summary(subject='%d' % neuron, method='I'))
                evaluations.append(Model2.summary(subject='%d' % neuron, method='II'))

            data = pd.DataFrame(evaluations)
            data.to_csv(evaluation_filename)

        return data

    def report(self):
        """
        Evaluating both models for all neurons, load from csv if already exist.
        :return: handle for text/YAML file
        """
        report_filename = os.path.join(self.results_directory, 'report.yaml')
        report = open(report_filename, "w")
        yaml.dump(self.metadata(), report)
        return report