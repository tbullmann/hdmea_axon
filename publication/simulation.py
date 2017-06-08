from __future__ import division

from brian2 import *
import matplotlib.pyplot as plt
from itertools import product   # must import product from itertools after brian2
from tqdm import *
import os
import pickle
from pprint import pprint

from data import Experiment
import networkx as nx
from hana.plotting import plot_network, plot_neuron_points, plot_neuron_id
from hana.misc import unique_neurons

import logging
logging.basicConfig(level=logging.DEBUG)

SIMULATED_INTERVAL = 300  # 300 s == 5 min
# SIMULATED_NETWORK = 'same size and connectivity'
# SIMULATED_NETWORK = '50 neurons same connectivity'
SIMULATED_NETWORK = 'same delays'
# SIMULATED_NETWORK = 'test'

class Simulation(Experiment):

    def __init__(self, culture, data_base_dir='temp/sim', temp_base_dir='temp/sim'):
        self.culture = culture
        self.data_base_dir = data_base_dir
        self.temp_base_dir = temp_base_dir
        self.sub_dir = 'culture%d' % culture
        self.sim_directory = os.path.join(self.data_base_dir, self.sub_dir)
        if not os.path.exists(self.sim_directory):
            os.makedirs(self.sim_directory)
        self.events_filename = os.path.join(self.sim_directory, 'events.h5')
        self.delay_filename = os.path.join(self.sim_directory, 'delays.h5')

        self.results_directory = os.path.join(self.temp_base_dir, self.sub_dir)
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

    def parameters(self):
        structural_strength, structural_delay, _, _, _, _ = Experiment(self.culture).networks()

        if SIMULATED_NETWORK == 'same size and connectivity':
            # simulate Erdoes Renyi type random network with same number of neurons and connections
            M = len(structural_delay.keys())   # number of connections
            n = len(unique_neurons(structural_delay))  # number of neurons
            parameters = dict(T=SIMULATED_INTERVAL,
                              N=n,
                              k=M/(n-1),
                              )

        elif SIMULATED_NETWORK == '50 neurons same connectivity':
            # simulate Erdoes Renyi type random network with 50 number and same connection density
            M = len(structural_delay.keys())  # number of connections
            n = len(unique_neurons(structural_delay))  # number of neurons
            p = M / (n * (n - 1))
            N = 50
            parameters = dict(T=SIMULATED_INTERVAL,
                              N=N,
                              k=p * N,
                              )

        elif SIMULATED_NETWORK == 'same delays':
            # simualate network using the original axonal delays for each neuron pair
            M = len(structural_delay.keys())   # number of connections
            n = len(unique_neurons(structural_delay))  # number of neurons
            parameters = dict(T=SIMULATED_INTERVAL,
                              N=max(unique_neurons(structural_delay))+1,  # make more neurons but use only those connected
                              k=M/(n-1),
                              overlap=structural_strength,
                              tau_axon=structural_delay,
                              )

        else:  # standard for testing
            parameters = dict(T=SIMULATED_INTERVAL,
                              N=50,
                              k=0.3 * 50,
                              )

        logging.info('Simulate network with %s as culture %d' % (SIMULATED_NETWORK, self.culture) )
        # pprint(parameters)

        # TODO save paramters to JSON and maybe load from there

        return parameters

    def run_and_save(self):
        structural_delay, timeseries = simulate_network(**self.parameters())
        pickle.dump(structural_delay, open(self.delay_filename, 'wb'))
        pickle.dump(timeseries, open(self.events_filename, 'wb'))
        logging.info('Saved connectivity and timeseries')

    def timeseries(self):
        """
        Load time series obtained by simulation.
        :return: time series indexed by neurons
        """
        if not os.path.isfile(self.events_filename):
            self.run_and_save()
        timeseries = pickle.load(open(self.events_filename, 'rb'))
        return timeseries

    def structural_network(self):
        """
        Load structural network used for simulation
        :return: structural connectivity indexed by neuron pairs
        """
        if not os.path.isfile(self.delay_filename):
            self.run_and_save()
        logging.info('Using structural connectivity from simulation!')
        structural_delay = pickle.load(open(self.delay_filename, 'rb'))
        structural_strength = dict()  # empty, because no overlap defined
        return structural_delay, structural_strength

def simulate_network (T = 10, N = 50, k = 15, refractory=5*ms, tau_re = 1300 * ms, U = 0.05, g = 40,
                      tau_axon = 2 * ms, overlap=None, tau_synapse = 3 * ms, tau_mem = 20 * ms):
    """
    Simulate a neuronal network, either random or based on actual connectivity.
    :param T: simulation interval in seconds (default:10)
    :param N: number of neurons
    :param k: connections per neuron
    :param refractory: refractory time (default: 5*ms)
    :param tau_axon: axonal delay in ms indexed by neuron pair; default: 2*ms, used for all connections
    :param overlap: overlap betwen axon and dendrite indexed by neuron pair; default None, then scaling 1/k used
    :param tau_synapse: synaptic delay in ms (default: 3*ms)
    :param tau_mem: membrane time constant, typically in the range of 20 to 60*ms (default: 20*ms)
    :param tau_re: timeconstant for the recovery of synaptic resources (default: 1300*ms)
    :param U: fraction of utilized synaptic resources per pre-synaptic spike (default: 0.10)
    :param g: synaptic strength as scaled conductance (default: 40)
    :return: axonal_delays: axonal delays indexed by pre- and post-synaptic neuron pair
    :return: timeseries: events indexed by neuron
    """

    start_scope()

    logging.info('Define neurons, synapses, connectivity and delays')
    eqs = '''
          dv/dt = (a-v)/tau_mem + sigma*xi*tau_mem**-0.5 : 1 (unless refractory)
          a : 1
          '''
    G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', refractory=refractory, method='euler')
    G.a = 0.5

    state_eqs = '''
        w : 1
        du/dt = (1-u)/tau_re : 1 (clock-driven)
        '''
    update_eqs = '''
        v_post += w * g * U * u
        u -= u * U
        '''
    S = Synapses(G, G, state_eqs, on_pre=update_eqs, method='euler')

    if type(tau_axon) is dict:   # simulate network with given axonal delays for each neuron pair
        for pre, post in tau_axon:
            S.connect(i=pre, j=post)
            if type(overlap) is dict:   # strength proportional to (normalised) overlap
                S.w[pre, post] = overlap[pre, post] / np.sum(overlap[i,j] for i,j in overlap.keys() if j==post)
            else:
                S.w[pre, post] = 1 / k
            S.delay[pre, post] = tau_synapse + tau_axon[pre, post] * ms
        for neuron in range(N):
            if not neuron in unique_neurons(tau_axon):
                G.a[neuron] = 0  # inactivate neuron
    else:   # simulate Erdoes Renyi type random network
        S.connect(condition='i!=j', p=k / N)
        S.w = 1 / k
        S.delay = tau_synapse + tau_axon * rand(size(S.w))

    G.v = 'rand()'

    logging.info('Simulate %d s' % T)
    spikemon = SpikeMonitor(G)
    for _ in tqdm(range(T)):
        run(1 * second)

    logging.info('Report results')
    axonal_delays = {(pre, post): float((S.delay[pre, post] - tau_synapse) / ms)
                     for pre, post in product(range(0, N+1), repeat=2)
                     if S.delay[pre, post] / ms > 0}
    t = spikemon.t / second
    i = spikemon.i
    timeseries = {neuron:t[i==neuron] for neuron in np.unique(i)}

    return axonal_delays, timeseries


def logging_spont_firing(a, tau_mem):
    T = np.log(a / (a - 1)) * tau_mem
    logging.info("%d active neurons with a period of %f - %f (median=%f) s" % (
    np.count_nonzero(np.isfinite(T)), np.nanmin(T), np.nanmax(T), np.nanmedian(T)))


def springs(structural_delay):
    G = nx.DiGraph()
    G.add_edges_from(structural_delay)
    pos = nx.spring_layout(G)
    max_neuron = max(pos.keys())
    x = np.zeros(max_neuron + 1)
    y = np.zeros(max_neuron + 1)
    for neuron in sorted(pos):
        x[neuron] = pos[neuron][0]
        y[neuron] = pos[neuron][1]
    return np.rec.fromarrays((x, y), dtype=[('x', 'f4'), ('y', 'f4')])


def test_simulate_network():

    # Getting data
    # structural_delay, timeseries = simulate_network()
    structural_delay, _ = Simulation(1).structural_network()
    timeseries = Simulation(1).timeseries()

    # Plotting
    plt.figure('network')
    pos = springs(structural_delay)
    ax1 = subplot(111)
    plot_network(ax1, structural_delay, pos, color='blue')
    plot_neuron_points(ax1, unique_neurons(structural_delay), pos)
    plot_neuron_id(ax1, unique_neurons(structural_delay), pos)

    plt.figure('spike')
    for neuron in timeseries:
        t = timeseries[neuron]
        plt.plot(t, neuron * np.ones_like(t), '.')
    xlabel(r'$t$ [s]')
    ylabel('Neuron index')
    plt.show()


if __name__ == "__main__":
    test_simulate_network()