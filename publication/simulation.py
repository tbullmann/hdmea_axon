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

SIMULATED_INTERVAL = 120  # 120s
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
        _, structural_delay, _, _, _, _ = Experiment(self.culture).networks()

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
        if SIMULATED_NETWORK == 'same delays':
            logging.info('Using structural connectivity from simulation!')
            structural_strength, structural_delay, _, _, _, _ = Experiment(self.culture).networks()
        else:
            logging.info('Using structural connectivity from simulation!')
            structural_delay = pickle.load(open(self.delay_filename, 'rb'))
            structural_strength = dict()  # empty, because no overlap defined
        return structural_delay, structural_strength


def simulate_network(T=10, N=50, k=15, tau_axon=2*ms):
    """"""
    start_scope()

    # --- Neurons ---
    logging.info('Define neurons')
    delta_a = 1e-9  # stable for e-11 .. e-9
    eqs = '''
        dv/dt = (a-v)/tau_mem: 1
        a : 1
        '''
    G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', method='euler', refractory=1 * ms)
    G.a = linspace(1 - 1 * delta_a, 1 + delta_a, N)  # T = log(a/(a-1))*tau_mem
    tau_mem = 30 * ms  # membrane time constant of different neurons typically range from 20 to 60 ms

    # --- Synapses ---
    logging.info('Define connectivity')
    tau_in = 10 * ms  # this should be around 2 ms
    tau_re = 130 * tau_in
    U = 0.10
    g = 20
    state_eqs = '''
        w : 1
        x = 1 - y - z : 1
        dy/dt = -y/tau_in : 1 (clock-driven)
        dz/dt = y/tau_in -z/tau_re : 1 (clock-driven)
        '''
    update_eqs = '''
        v_post += w * g * U * x
        y += U * x
        x -= U * x
        '''
    S = Synapses(G, G, state_eqs, on_pre=update_eqs, method='euler')

    if type(tau_axon) is dict:   # simulate network with given axonal delays for each neuron pair
        synapse_delay = 1 * ms
        for pre, post in tau_axon:
            S.connect(i=pre, j=post)
            S.w[pre, post] = 1 / k
            S.delay[pre, post] = synapse_delay + tau_axon[pre, post] * ms
        for neuron in range(N):
            if not neuron in unique_neurons(tau_axon):
                G.a[neuron] = 0  # inactivate neuron
    else:   # simulate Erdoes Renyi type random network
        S.connect(condition='i!=j', p=k / N)
        S.w = 1 / k
        synapse_delay = 1 * ms
        S.delay = synapse_delay + tau_axon * rand(size(S.w))



    # --- Initialization of variables
    G.v = 'rand()'
    S.y = 'rand()'
    S.z = 'rand()'

    # --- Simulation ----
    logging.info('Simulate %d s' % T)
    spikemon = SpikeMonitor(G)
    for _ in tqdm(range(T)):
        run(1 * second)

    # --- Report results ---
    logging.info('Report results')
    axonal_delays = {(pre, post): float((S.delay[pre, post] - synapse_delay) / ms)
                     for pre, post in product(range(0, N+1), repeat=2)
                     if S.delay[pre, post] / ms > 0}
    t = spikemon.t / second
    i = spikemon.i
    timeseries = {neuron:t[i==neuron] for neuron in np.unique(i)}

    return axonal_delays, timeseries


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

    # axonal_delays, timeseries = simulate_network()

    # Getting data

    # structural_delay, _ = Simulation(1).structural_network()
    timeseries = Simulation(1).timeseries()

    # Plotting
    # pos = springs(structural_delay)
    # ax1 = plt.subplot(121)
    # plot_network(ax1, structural_delay, pos, color='blue')
    # plot_neuron_points(ax1, unique_neurons(structural_delay), pos)
    # plot_neuron_id(ax1, unique_neurons(structural_delay), pos)
    ax2 = subplot(111)
    for neuron in timeseries:
        t = timeseries[neuron]
        ax2.plot(t, neuron * np.ones_like(t), '.k')
    xlabel(r'$t$ [s]')
    ylabel('Neuron index')
    plt.show()


if __name__ == "__main__":
    test_simulate_network()