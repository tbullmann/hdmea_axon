from hana.h5dict import save_dict_to_hdf5
from publication.plotting import FIGURE_ARBORS_FILE, FIGURE_EVENTS_FILE, FIGURE_ELECTRODES_MATFILE

import numpy as np
import scipy.io as sio
import logging
logging.basicConfig(level=logging.DEBUG)


HIDENS_ELETRODES_FILE = 'data/hidens_electrodes.h5'


def load_events(filename):
    """Loads spiking events from matlab file with frame(=20000*ts) and neuron index in all_events"""
    data = sio.loadmat(filename)["all_events"]
    timestamp = data[0][0][0][0].astype('f8') / 20000.0
    neuron = data[0][0][1][0]
    array = np.rec.fromarrays((timestamp, neuron), dtype=[('time', 'f8'), ('id', 'i4')])
    array.sort(order=['time'])
    return array


def events_to_timeseries(events):
    """Converts events consisting of tupels (time, neuron) to a dictonary with neuron as key and timeseries as value"""
    times, neurons = zip(*events)
    unique_neurons = np.unique(neurons)
    logging.info("uniques neurons: ", unique_neurons)
    timeseries = dict([(neuron, np.array(times)[neurons == neuron]) for neuron in unique_neurons])
    return timeseries


def load_positions(filename):
    """Loads electrode positions"""
    data = sio.loadmat(filename)["hidens_electrodes"]
    x = data[0][0][0][0]
    y = data[0][0][1][0]
    return np.rec.fromarrays((x, y), dtype=[('x', 'f4'), ('y', 'f4')])


def load_neurites(filename):
    """Loads delay map into dictionary"""
    data = sio.loadmat(filename)["arbors"]
    delay = {}
    positive_peak = {}
    for record in data[0]:
        delay[record[0][0][0][0][0]] = record[0][0][1][0]
        positive_peak[record[0][0][0][0][0]] = record[0][0][2][0]

    # Indicies for neurons in events starting at 0, in delays and positive peak from 1 (!)
    # TODO: Fix this BIG BUG in the matlab export script (?) and delete the quick fix below
    delay = {key - 1: value for key, value in delay.items()}
    positive_peak = {key - 1: value for key, value in positive_peak.items()}

    logging.info("Delays: ", delay)

    return delay, positive_peak


def convert_matlab_events_to_hdf5_timeseries():
    """Convert events exported for trigger electrodes to timeseries indexed by neuron ID"""
    trigger_electrode, _, _, _ = load_neurites('../publication/' + FIGURE_ARBORS_FILE)
    events = load_events('../publication/' + FIGURE_EVENTS_FILE)
    timeseries = events_to_timeseries(events)
    print (trigger_electrode.keys())
    print (timeseries.keys())
    # Maybe convert neuron, trigger_electrode from numpy array to int in but the problem goes back to
    # extract_neurites --> structure.extract_all_compartments --> structure.load_traces --> matlab.load_traces so it shoudl
    timeseries = {int(neuron): timeseries[int(trigger_electrode[int(neuron)])] for neuron in trigger_electrode.keys()}
    print (timeseries.keys())
    save_dict_to_hdf5(timeseries, '../publication/data/events.h5')


def convert_matlab_positions_to_hdf5_positions(mat_filename=FIGURE_ELECTRODES_MATFILE, hdf5_filename=HIDENS_ELETRODES_FILE):
    pos_named_tuple = load_positions(mat_filename)
    pos = {'x':pos_named_tuple.x, 'y':pos_named_tuple.y}
    save_dict_to_hdf5(pos, hdf5_filename)
