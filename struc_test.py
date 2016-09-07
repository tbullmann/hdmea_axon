from itertools import product
from matplotlib import pyplot as plt
import numpy as np
from mio import load_neurites, load_positions
from plot_network import set_axis_hidens, plot_neuron_points, plot_neuron_id, highlight_connection
from plot_network import plot_network
import pickle
import logging
logging.basicConfig(level=logging.DEBUG)



cm_axon = plt.cm.ScalarMappable(cmap=plt.cm.summer, norm=plt.Normalize(vmin=0, vmax=2))
cm_dendrite = plt.cm.ScalarMappable(cmap=plt.cm.gray_r, norm=plt.Normalize(vmin=0, vmax=50))


def plot_neurite(ax, cm, z, pos, alpha=1, thr=0):
    index = np.isfinite(z) & np.greater(z,thr)
    x = pos.x[index]
    y = pos.y[index]
    c = cm.to_rgba(z[index])
    ax.scatter(x, y, 18, c, marker='h', edgecolor='none', alpha=alpha)

def plot_axon(ax, pos, z):
    plot_neurite(ax, cm_axon, z, pos)

def plot_dendrite(ax, pos, z, thr=10):
    plot_neurite(ax, cm_dendrite, z, pos, alpha=0.8, thr=thr)


def test():
    axon_delay, dendrite_peak = load_neurites ('data/hidens2018at35C_arbors.mat')
    pos = load_positions('data/hidens_electrodes.mat')

    print len(axon_delay)
    print axon_delay.keys()

    # neuron indicies are 6272, 9861, 7050, 5396, 3094, 6295, 7965, 6533, 3241, 3606, 2874, 5697, 10052, 8901, 10569,
    # 7375, 10964, 5978, 7903, 9953, 7014, 6249, 4973, 5243, 8061]


    pre = 3241
    post =4973

    ax = plt.subplot(111)

    z = axon_delay[pre]
    print np.nanmax(z)

    plot_axon(ax, pos, z)

    # for neuron in dendrite_peak:
    #     z = dendrite_peak[neuron]
    #     plot_dendrite(ax,pos,z)

    z = dendrite_peak[post]
    plot_dendrite(ax, pos, z)
    plot_neuron_points(ax, axon_delay, pos)
    plot_neuron_id(ax, axon_delay, pos)
    highlight_connection(ax,(pre,post),pos)

    set_axis_hidens(ax,pos)


    plt.show()

def test_plot_all_axonal_fields():
    """Display axonal field for each neuron, good examples are 3606, 9953, 3094"""
    axon_delay, dendrite_peak = load_neurites ('data/hidens2018at35C_arbors.mat')
    pos = load_positions('data/hidens_electrodes.mat')
    for neuron in axon_delay :
        ax=plt.subplot(111)
        plot_axon(ax,pos,axon_delay[neuron])
        set_axis_hidens(ax,pos)
        ax.set_title ('axon for neuron %d' % neuron)
        plt.show()

def test_plot_all_dendritic_fields(presynaptic_neuron):
    """Display dendritic field for each neuron,
    good examples are 3606->7375,5243,4973,not 6272,6295,8061; 9953->not3606,7375,4973,5243;3094->3606, not7375,5243,4973"""
    axon_delay, dendrite_peak = load_neurites ('data/hidens2018at35C_arbors.mat')
    pos = load_positions('data/hidens_electrodes.mat')
    for postsynaptic_neuron in dendrite_peak :
        ax=plt.subplot(111)
        plot_axon(ax, pos, axon_delay[presynaptic_neuron])
        plot_dendrite(ax,pos,dendrite_peak[postsynaptic_neuron])
        set_axis_hidens(ax,pos)
        ax.set_title ('axon for neuron %d, dendrite for neuron %d' % (presynaptic_neuron, postsynaptic_neuron))
        plt.show()

def test_plot_overlap():
    """Plot overlap between axonal and dendritic fields of a presumably pre- and post-synaptic neuron pair"""
    axon_delay, dendrite_peak = load_neurites ('data/hidens2018at35C_arbors.mat')
    pos = load_positions('data/hidens_electrodes.mat')
    presynaptic_neuron = 3606
    postsynaptic_neuron = 4973  # 5243
    delay = 2.3

    ax=plt.subplot(111)

    plot_neuron_pair(ax, pos, axon_delay, dendrite_peak, delay, postsynaptic_neuron, presynaptic_neuron)

    set_axis_hidens(ax,pos)
    ax.set_title ('neuron pair %d $\longrightarrow$ %d' % (presynaptic_neuron, postsynaptic_neuron))
    plt.show()


def test_estimate_overlap():
    axon_delay, dendrite_peak = load_neurites ('data/hidens2018at35C_arbors.mat')
    pos = load_positions('data/hidens_electrodes.mat')

    presynaptic_neuron = 3606
    postsynaptic_neuron = 4973  # 5243
    thr_peak = 5
    thr_overlap = 0.05

    ratio, delay = find_overlap(axon_delay, dendrite_peak, presynaptic_neuron, postsynaptic_neuron, thr_peak, thr_overlap)

    ax1 = plt.subplot(121)
    plot_neuron_pair(ax1, pos, axon_delay, dendrite_peak, delay, postsynaptic_neuron, presynaptic_neuron)
    set_axis_hidens(ax1,pos)
    ax1.set_title ('neuron pair %d $\longrightarrow$ %d' % (presynaptic_neuron, postsynaptic_neuron))

    ax2 = plt.subplot(122)
    all_ratio, all_delay = all_overlaps(axon_delay,dendrite_peak)
    plot_network (ax2, all_delay, pos)
    set_axis_hidens(ax2,pos)
    ax2.set_title ('structural connectivity graph')
    plt.show()


def find_overlap(axon_delay, dendrite_peak, presynaptic_neuron, postsynaptic_neuron, thr_peak=10, thr_overlap=0.10):
    delay = np.nan
    overlap_ratio = np.nan
    axon_field = np.greater(axon_delay[presynaptic_neuron], 0)
    dendritic_field = np.greater(dendrite_peak[postsynaptic_neuron], thr_peak)
    overlap = np.logical_and(axon_field, dendritic_field)
    dendrite_size = sum(dendritic_field)
    if dendrite_size > 0:
        overlap_size = sum(overlap)
        overlap_ratio = float(overlap_size) / dendrite_size
        if thr_overlap < overlap_ratio:
            delay = np.mean(axon_delay[presynaptic_neuron][overlap])
            logging.debug('overlap = %1.2f with mean delay = %1.1f [ms]' % (overlap_ratio, delay))
        else:
            logging.debug('overlap = %1.2f too small, no delay assigned' % overlap_ratio)
    return overlap_ratio, delay


def all_overlaps (axon_delay, dendrite_peak, thr_peak=10, thr_overlap=0.10):
    """Compute overlaps"""
    all_overlap_ratios, all_axonal_delays = [], []
    for pair in product(axon_delay, repeat=2):
        presynaptic_neuron, postsynaptic_neuron = pair
        if presynaptic_neuron<>postsynaptic_neuron:
            logging.debug('neuron %d -> neuron %d:' % pair)
            ratio, delay = find_overlap(axon_delay, dendrite_peak, presynaptic_neuron, postsynaptic_neuron,
                                        thr_peak=thr_peak, thr_overlap=thr_overlap)
            if np.isfinite(delay):
                all_overlap_ratios.append((pair, ratio))
                all_axonal_delays.append((pair, delay))
    return dict(all_overlap_ratios), dict(all_axonal_delays)


def plot_neuron_pair(ax, pos, axon_delay, dendrite_peak, delay, postsynaptic_neuron, presynaptic_neuron):
    plot_axon(ax, pos, axon_delay[presynaptic_neuron])
    plot_dendrite(ax, pos, dendrite_peak[postsynaptic_neuron])
    highlight_connection(ax, (presynaptic_neuron, postsynaptic_neuron), pos,
                         annotation_text=' %1.1f ms' % delay)

def figure10_prepare_data():
    axon_delay, dendrite_peak = load_neurites ('data/hidens2018at35C_arbors.mat')

    resolution = 3
    alpha = np.float64(range(-5*resolution,5*resolution+1))/resolution
    thresholds_overlap = list((2**alpha)/(2**alpha+1))
    resolution = resolution*2
    thresholds_peak  = list(2**(np.float(exp+1)/resolution)-1 for exp in range(5*resolution+1))

    print 'total', len(thresholds_peak), 'thresholds for peak ', thresholds_peak
    print 'total', len(thresholds_overlap), 'thresholds for overlap = ', thresholds_overlap

    networks = []

    for thr_peak in thresholds_peak:
        for thr_overlap in thresholds_overlap:
            print 'Connections for peak > %1.1f mV and overlap > = %1.2f' % (thr_peak, thr_overlap)
            all_ratio, all_delay = all_overlaps(axon_delay, dendrite_peak, thr_peak=thr_peak, thr_overlap=thr_overlap)
            k = len(all_ratio)
            print 'Network connection k = %d' % k
            networks.append(((thr_peak, thr_overlap),{'overlap ratio': all_ratio, 'delay': all_delay}))

    print 'Finished exploring %d different parameter sets' % len(networks)

    pickle.dump(dict(networks), open('temp/struc_networks_for_thr_peak_thr_overlap_hidens2018.p', 'wb'))

    print 'Saved data'



# test_plot_all_axonal_fields()
# test_plot_all_dendritic_fields(3606)
# test_plot_overlap()
# test_estimate_overlap()
figure10_prepare_data()
