from __future__ import division
from publication.plotting import plot_parameter_dependency

from numpy import linspace, zeros, ones
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
import os, pickle, logging
logging.basicConfig(level=logging.DEBUG)


# Data preparation


def compare_structural_and_functional_networks():

    # Loading networks
    logging.info('Loading Structural networks:')
    structural_networks = pickle.load(open('temp/structural_networks.p', 'rb'))
    thresholds_peak, thresholds_overlap = (list(sorted(set(index)) for index in zip(*list(structural_networks))))
    logging.info('%d thresholds_peak x %d thresholds_overlap' % (len(thresholds_peak), len(thresholds_overlap)))

    logging.info('Loading Functional networks:')
    functional_networks = pickle.load(open('temp/functional_networks.p', 'rb'))
    factors, thresholds, directions = (list(sorted(set(index)) for index in zip(*list(functional_networks))))
    logging.info('%d randomization factors x %d z score thresholds x %d' % (len(factors),len(thresholds),len(directions)))

    # Preparing output
    k_intersect = zeros((len(thresholds_peak), len(factors)))
    i_jacc = zeros((len(thresholds_peak), len(factors)))
    i_struc = zeros((len(thresholds_peak), len(factors)))
    i_func = zeros((len(thresholds_peak), len(factors)))
    corr_delay = zeros((len(thresholds_peak), len(factors)))
    p_delay = ones((len(thresholds_peak), len(factors)))
    corr_strength = zeros((len(thresholds_peak), len(factors)))
    p_strength = ones((len(thresholds_peak), len(factors)))

    thr_peak = thresholds_peak[14]
    direction = directions[0]
    logging.info('Fixed threshold_peak=%df mV, direction=%s' % (thr_peak, direction))
    factors.sort(reverse=True)

    for i, thr_overlap in enumerate(thresholds_overlap):
        logging.info('Comparing %dth structural network with thr_peak =%f mV, thr_overlap=%f' % (i, thr_peak, thr_overlap))
        structural_delay = structural_networks[(thr_peak, thr_overlap)]['delay']
        structural_strength = structural_networks[(thr_peak, thr_overlap)]['overlap ratio']
        for j, (factor, thr) in enumerate(zip(factors,thresholds)):
            logging.info('Comparing %dth functional network with factor =%f, thr=%f' % (j, factor, thr))
            functional_delay = functional_networks[(factor, thr, direction)]['peak_timelag']
            functional_strength = functional_networks[(factor, thr, direction)]['peak_score']

            # Get connections (as keys), compute their intersection and union
            structural_keys = set(structural_delay.keys())
            functional_keys = set(functional_delay.keys())
            intersection_keys = structural_keys & functional_keys  # '&' operator is used for set intersection
            union_keys = structural_keys | functional_keys  # '|' operator is used for set union

            # Calculate size of intersections and Jaccard, structural and functional indices
            k_intersect[i,j] = len(intersection_keys)
            if len(union_keys)>0: i_jacc[i,j] = len(intersection_keys)/len(union_keys)
            if len(structural_keys)>0: i_struc[i,j] = len(intersection_keys)/len(structural_keys)
            if len(functional_keys)>0: i_func[i,j] = len(intersection_keys)/len(functional_keys)

            # Calculate correlations for delays and strength using the intersection_keys
            # scipy.stats.pearsonr(x, y) returns Pearson correlation coefficient and
            #   the 2-tailed p-value for Pearson's correlation coefficient, which is however it is not entirely
            #   reliable for small datasets (N<500)
            if len(intersection_keys)>2:  # Pearson coefficient is always +/- 1 for only 2 data points(!)
                logging.info('Axonal delay ~ Spike time lag: ')
                delay_pairs = ([(structural_delay[c], functional_delay[c]) for c in intersection_keys])
                logging.info(delay_pairs)
                pearson_delay_pairs, p_delay_pairs = pearsonr(*zip(*delay_pairs))
                logging.info('Pearson correlation coefficient = %f (p=%f, N=%d)'
                             % (pearson_delay_pairs, p_delay_pairs, len(intersection_keys)))
                corr_delay[i, j] = pearson_delay_pairs
                p_delay[i,j] = p_delay_pairs
                logging.info('Axon dendrite overlap ~ z score of spike timing: ')
                strength_pairs = ([(structural_strength[c], functional_strength[c]) for c in intersection_keys])
                logging.info(strength_pairs)
                pearson_strength_pairs, p_strength_pairs = pearsonr(*zip(*strength_pairs))
                logging.info('Pearson correlation coefficient = %f (p=%f, N=%d)'
                             % (pearson_strength_pairs, p_strength_pairs, len(intersection_keys)))
                corr_strength[i, j] = pearson_strength_pairs
                p_strength[i, j] = p_strength_pairs
            else:
                logging.info('Nothing to calculate. Correlation coefficients set to 0, p value set to 1.')

    pickle.dump((thr_peak, factors, thresholds_overlap,
                 k_intersect, i_jacc, i_struc, i_func, corr_delay, p_delay, corr_strength, p_strength),
                open('temp/matching_networks.p', 'wb'))


def scale_and_label(ax):
    ax.set_yscale('log')
    ax.set_xlabel('Structure parameters: $\phi = x, \\rho=5mV$')
    ax.set_ylabel('Function parameters: $\sigma = y, \zeta = 10^{1.8}/y$')


def figure11():
    plt.figure('Figure 11', figsize=(12,12))

    thr_peak, factors, thresholds_overlap, \
    k_intersect, i_jacc, i_struc, i_func, corr_delay, p_delay, corr_strength, p_strength \
        = pickle.load(open('temp/matching_networks.p', 'rb'))

    logging.info('Plotting results for functional and structural networks')

    ax = plt.subplot(221)
    plot_parameter_dependency(ax, k_intersect, thresholds_overlap, factors, levels=(5, 10, 20, 50, 100, 250, 500))
    ax.set_title('number of intersecting edges, $k_{Intersection}$')
    scale_and_label(ax)

    ax = plt.subplot(222)
    plot_parameter_dependency(ax, i_jacc, thresholds_overlap, factors, fmt='%1.1f')
    ax.set_title('Jaccard index, $i_{Jaccard}$')
    scale_and_label(ax)

    ax = plt.subplot(223)
    plot_parameter_dependency(ax, i_struc, thresholds_overlap, factors, fmt='%1.1f')
    ax.set_title('Structural index, $i_{Structural}$')
    scale_and_label(ax)

    ax = plt.subplot(224)
    plot_parameter_dependency(ax, i_func, thresholds_overlap, factors, fmt='%1.1f')
    ax.set_title('Functional index, $i_{Functional}$')
    scale_and_label(ax)

    plt.show()


def figure11b():
    plt.figure('Figure 11b', figsize=(12,12))

    thr_peak, factors, thresholds_overlap, \
    k_intersect, i_jacc, i_struc, i_func, corr_delay, p_delay, corr_strength, p_strength \
        = pickle.load(open('temp/matching_networks.p', 'rb'))

    logging.info('Plotting results for functional and structural networks')

    r_levels = linspace(-1,1,11)
    p_levels = (0, 0.01, 0.05, 0.50, 1)

    ax = plt.subplot(221)
    plot_parameter_dependency(ax, corr_delay, thresholds_overlap, factors, fmt='%1.1f', levels=r_levels)
    ax.set_title('Correlation of delays, $r_{Pearson}$')
    scale_and_label(ax)

    ax = plt.subplot(222)
    plot_parameter_dependency(ax, p_delay, thresholds_overlap, factors, fmt='%1.2f', levels=p_levels)
    ax.set_title('Correlation of delays, $p_{Pearson}$')
    scale_and_label(ax)

    ax = plt.subplot(223)
    plot_parameter_dependency(ax, corr_strength, thresholds_overlap, factors, fmt='%1.1f', levels=r_levels)
    ax.set_title('Correlation of strength, $r_{Pearson}$')
    scale_and_label(ax)

    ax = plt.subplot(224)
    plot_parameter_dependency(ax, p_strength, thresholds_overlap, factors, fmt='%1.2f', levels=p_levels)
    ax.set_title('Correlation of strength, $p_{Pearson}$')
    scale_and_label(ax)

    plt.show()


if not os.path.isfile('temp/matching_networks.p'): compare_structural_and_functional_networks()
figure11()
figure11b()

# x = (1, 2)
# y = (10, 20, 30)
# z = ones((len(x), len(y)))
# print z.shape
#
# ax = plt.subplot(111)
# plot_parameter_dependency(ax, z, x,y, fmt='%1.1f')
# plt.show()


# import numpy
# list1 = (1, 2, 3, 0)
# list2 = (2, 3, 5, 1)
# print(numpy.corrcoef(list1, list2)[0,1])