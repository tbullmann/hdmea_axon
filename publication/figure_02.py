from hana.matlab import load_traces
from publication.plotting import FIGURE_NEURON_FILE

import logging
logging.basicConfig(level=logging.DEBUG)

from matplotlib import pyplot as plt


# Testing code

def testing_load_traces():
    V, t, x, y, trigger, neuron = load_traces(FIGURE_NEURON_FILE)


# Final figure 2

def figure02():
    plt.figure('Figure 11', figsize=(12,12))
    V, t, x, y, trigger, neuron = load_traces(FIGURE_NEURON_FILE)
    Vtrigger = V[int(trigger)]
    V = V[range(trigger-10,trigger)+range(trigger+1,trigger+11)] # for testing

    ax = plt.subplot(221)
    print V.shape
    print t
    ax.plot(V.T,'-', color='gray')
    plt.show()


figure02()
