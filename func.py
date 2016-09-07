"""Func provides functional connectivity"""
from itertools import product, groupby
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)


def timelag_by_for_loop (timeseries1, timeseries2):
    """Returns for each event in the first time series the time lags for the event in the second time series
    that precedes, succeeds. Both time series must be sorted in increasing values."""
    preceding_time_lags = []
    succeeding_time_lags = []
    for time1 in timeseries1:
        preceding_time_lags.append(next((time2 - time1 for time2 in reversed(timeseries2) if time2 < time1), []))
        succeeding_time_lags.append(next((time2 - time1 for time2 in timeseries2 if time2 > time1), []))
    return np.sort(np.hstack(preceding_time_lags + succeeding_time_lags))


def sawtooth(timeseries, dtype = np.float32):
    """Sawtooth function expressing the time lag to the next event in the timeseries."""
    epsilon = np.finfo(dtype).eps
    gaps = np.diff(timeseries)
    x = np.column_stack((timeseries[0:-1], timeseries[1:] - epsilon)).flatten()
    y = np.column_stack((gaps, np.zeros_like(gaps))).flatten()
    return [x, y]


def timelag_by_sawtooth (timeseries1, timeseries2):
    """Returns for each event in the first time series the time lags for the event in the second time series
    that precedes, succeeds. Both time series must be sorted in increasing values. Faster than using the for loop"""
    preceding_time_lags = - np.interp(np.flipud(-timeseries1), *sawtooth(-np.flipud(timeseries2)), left=np.nan, right=np.nan)
    succeeding_time_lags = np.interp(timeseries1, *sawtooth(timeseries2), left=np.nan, right=np.nan)
    time_lags =  np.sort(np.hstack([preceding_time_lags, succeeding_time_lags]))
    valid_time_lags = (np.ma.fix_invalid(time_lags))
    return np.ma.compressed(valid_time_lags)


timelag = timelag_by_sawtooth


def timelag_hist (timelags, min_timelag=-0.005, max_timelag=0.005, bin_n=100):
    bins = np.linspace(min_timelag, max_timelag, bin_n + 1, endpoint=True)
    return np.histogram(timelags, bins=bins)


def swap_intervals (timeseries, indicies):
    """Swap intervals between adjacent intervals indicated by indicies"""
    intervals = np.diff(timeseries)
    for index in indicies:
        intervals[index], intervals[index+1] = intervals[index+1], intervals[index]
    return np.hstack([timeseries[0], timeseries[0]+np.cumsum(intervals)])


def randomize_intervals_by_swapping (timeseries, factor):
    """Randomize timeseries by randomly swapping adjacent intervals, total factor times the length of timeseries"""
    length = len(timeseries)-1
    times = round(factor*length,0)
    indicies = np.random.randint(0,length-1,int(times))
    return swap_intervals(timeseries,indicies)


def randomize_intervals_by_gaussian (timeseries, factor):
    """Randomize timeseries by assuming indicies make a random walk with (+factor,-factor) of equal probability.
    Much faster than the old one."""
    gaps = np.diff(timeseries)
    length = len(gaps)
    new_positions = range(length) + np.random.normal(0, factor, length)
    index = np.argsort(new_positions)
    return timeseries[0] + np.hstack((0,np.cumsum(gaps[index])))


randomize_intervals = randomize_intervals_by_gaussian


def surrogate_timeseries (timeseries, n=10, factor=2):
    return [randomize_intervals(timeseries,factor=factor) for i in range(n)]


def timelag_standardscore(timeseries1, timeseries2, surrogates):
    """Returns timelags (midpoints of bins) and standard score as well as the counts from the orginal timeseries
    and mean and standard deviation for the counts from surrogate timeseries"""
    timeseries_hist, bins = timelag_hist(timelag(timeseries1, timeseries2))
    timelags = (bins[:-1] + bins[1:])/2
    surrogates_hist = np.vstack([timelag_hist(timelag(timeseries1, surrogate))[0] for surrogate in surrogates])
    surrogates_mean = surrogates_hist.mean(0)
    surrogates_std = np.std(surrogates_hist, 0)
    try: std_score = (timeseries_hist - surrogates_mean) / surrogates_std
    except ZeroDivisionError: pass
    return timelags, std_score, timeseries_hist, surrogates_mean, surrogates_std


def timeseries_to_surrogates(timeseries, n=10, factor=2):
    """Generating surrogate timeseries (this can take a while)"""
    timeseries_surrogates = dict([(key, surrogate_timeseries(timeseries[key], n=n, factor=factor)) for key in timeseries])
    return timeseries_surrogates


def all_timelag_standardscore (timeseries, timeseries_surrogates):
    """Compute standardscore time histograms"""
    all_std_score = []
    all_timeseries_hist = []
    for pair in product(timeseries, repeat=2):
        timelags, std_score, timeseries_hist,surrogates_mean, surrogates_std \
            = timelag_standardscore(timeseries[pair[0]], timeseries[pair[1]], timeseries_surrogates[pair[1]])
        logging.info ( "Timeseries %d->%d" % pair )
        all_std_score.append((pair, std_score))
        all_timeseries_hist.append((pair, timeseries_hist))
        # if logging.getLogger().getEffectiveLevel()==logging.DEBUG:
        #     plot_pair_func(timelags, timeseries_hist, surrogates_mean, surrogates_std, std_score,
        #                    "Timeseries %d->%d" % pair)
        #     plt.show()
    return timelags, dict(all_std_score), dict(all_timeseries_hist)


def find_peaks (x, y, thr=0):
    """Splits the argument x of the function y(x) into separate intervals [x_start, x_end] where y(x)>thr continuously
    and returns the y_max, x_max, x_start, x_end for each interval"""
    pieces = (zip(*list(g)) for k, g in groupby(enumerate(y), lambda iv: iv[1] > thr) if k)
    try: y_max, x_max, x_start, y_start = zip(*((max(v), x[i[np.argmax(v)]], x[i[0]], x[i[-1]]) for (i, v) in pieces))
    except: y_max, x_max, x_start, y_start = (), (), (), ()  # nothing above threshold
    return np.array(y_max), np.array(x_max), np.array(x_start), np.array(y_start)

def all_peaks (timelags, std_score_dict, thr=10, direction='both'):
    """Compute peaks"""
    all_score_max, all_timelag_max, all_timelag_start, all_timelag_end = [],[],[],[]
    for pair in std_score_dict:
        score_max, timelag_max, timelag_start, timelag_end = find_peaks(timelags, std_score_dict[pair], thr=thr)
        if direction=='reverse':
            timelag_max = -timelag_max
            timelag_start = - timelag_start
            timelag_end = - timelag_end
            pair = pair[::-1]
        if direction<>'both':  # if forward or reversed only, subset the peak descriptions
            index = timelag_max>0
            score_max = score_max[index]
            timelag_max = timelag_max[index]
            timelag_start = timelag_start[index]
            timelag_end = timelag_end[index]
        if len(score_max)>0:
            index_largest_peak = np.argmax(score_max)
            logging.info(("Timeseries %d->%d" % pair) +
                         (": max z = %f at %f s"  % (score_max[index_largest_peak], timelag_max[index_largest_peak])))
            all_score_max.append((pair, score_max[index_largest_peak]))
            all_timelag_max.append((pair, timelag_max[index_largest_peak]))
            all_timelag_start.append((pair, timelag_start[index_largest_peak]))
            all_timelag_end.append((pair, timelag_end[index_largest_peak]))
        else:
            logging.info(("Timeseries %d->%d" % pair) + ': no peak (above threshold)')
    return dict(all_score_max), dict(all_timelag_max), dict(all_timelag_start), dict(all_timelag_end)


