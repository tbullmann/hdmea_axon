import matplotlib.pyplot as plt
import numpy as np

from func import timeseries_to_surrogates, all_timelag_standardscore, timelag_by_for_loop, timelag_by_sawtooth, \
    timelag_hist, timelag, randomize_intervals_by_swapping, randomize_intervals_by_gaussian, surrogate_timeseries, \
    timelag_standardscore, plot_pair_func, find_peaks, all_peaks, swap_intervals
from mio import load_events

def test_swap_intervals ():
    timeseries = np.array([1,2,4,5,7,8,10])
    indicies = np.array([0, 4])
    print timeseries
    print indicies
    print swap_intervals(timeseries,indicies)

def test_timelag(n):
    timeseries1 = np.array(range(n))
    timeseries2 = timeseries1
    print timeseries1, " -> ", timeseries2
    print "timelags by for loop = ", timelag_by_for_loop(timeseries1, timeseries2)
    print "timelags by sawtooth = ", timelag_by_sawtooth(timeseries1, timeseries2)
    timeseries2 = timeseries1+0.1
    print timeseries1, " -> ", timeseries2
    print "timelags by for loop = ", timelag_by_for_loop(timeseries1, timeseries2)
    print "timelags by sawtooth = ", timelag_by_sawtooth(timeseries1, timeseries2)
    timeseries2 = timeseries1-0.2
    print timeseries1, " -> ", timeseries2
    print "timelags by for loop = ", timelag_by_for_loop(timeseries1, timeseries2)
    print "timelags by sawtooth = ", timelag_by_sawtooth(timeseries1, timeseries2)

def test_timelag_hist (n):
    timeseries1 = np.sort(np.random.rand(1, n))[0]
    timeseries2 = np.sort(np.random.rand(1, n))[0]
    print timelag_hist(timelag(timeseries1, timeseries2))[0]

def test_randomize_intervals (n, factor=2):
    timeseries = np.array(np.cumsum(range(n)))
    print "original timeseries    = ", timeseries
    print "gaps                   = ", np.diff(timeseries)
    print "randomized by swapping = ", np.diff(randomize_intervals_by_swapping(timeseries,factor))
    print "randomized by gaussian = ", np.diff(randomize_intervals_by_gaussian(timeseries,factor))

def test_surrogates (n):
    timeseries1 =  np.sort(np.random.rand(n+1))
    timeseries2 = np.sort(0.0002*np.random.rand(n+1) + np.copy(timeseries1)+0.002)
    surrogates = surrogate_timeseries(timeseries2, n=20)
    timelags, std_score, timeseries_hist, surrogates_mean, surrogates_std = timelag_standardscore(timeseries1,
                                                                                         timeseries2, surrogates)
    print "Score: Done."

    score_max, timelags_max, timelag_min, timelag_max = find_peaks (timelags, std_score, thr=10)
    print "Peaks: Done"

    if len(score_max)>0:
        print "peak score   =", score_max[0]
        print "peak timelag =", timelag_max[0]

    plot_pair_func(timelags, timeseries_hist, surrogates_mean, surrogates_std, std_score, 'Testing surrogate timeseries')
    plt.show()


def load_test_data():
    # choose experiment
    filename = "data/hidens2018at35C"
    # get events, covert to timeseries and make surrogates
    events = load_events(filename + "_events.mat")
    return events

def test_on_data():
    import mio
    events = load_test_data()
    timeseries = mio.events_to_timeseries(events)
    timeseries_surrogates = timeseries_to_surrogates(timeseries)
    timelags, std_score_dict, timeseries_hist_dict = all_timelag_standardscore(timeseries, timeseries_surrogates)

    import pickle
    pickle.dump((timelags, std_score_dict, timeseries_hist_dict),open( 'temp/standardscores_hidens2018.p','wb'))


    import matplotlib.pyplot as plt
    for pair in std_score_dict:
        plt.plot(timelags*1000, std_score_dict[pair])
    plt.show()

    score_max_dict, timelag_max_dict, timelag_start_dict, timelag_end_dict = all_peaks (timelags, std_score_dict, thr=10)

    print len(score_max_dict)


# test_timelag (5)
# test_timelag_hist(10000)
# test_randomize_intervals(10)
# test_surrogates(1000)
test_on_data()
