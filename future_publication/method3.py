import logging

import numpy as np
from matplotlib import pyplot as plt

from future_publication.flow import LucasKanade, interpolate
from hana.plotting import set_axis_hidens
from hana.recording import load_traces
from publication.plotting import FIGURE_NEURON_FILE_FORMAT

logging.basicConfig(level=logging.DEBUG)


def test_flows(neuron=5, coarse=True):
    filename = FIGURE_NEURON_FILE_FORMAT % neuron
    V, t, x, y, trigger, neuron = load_traces(filename)

    t *= 1000  # convert to ms
    timebefore = -4  # ms
    timeafter = 1  # ms

    gxB, gyB, gVB, gVsuccB, gvxB, gvyB, velocityB = velocity_by_LK(V, t, x, y, timebefore, coarse=coarse)
    gx, gy, gV, gVsucc, gvx, gvy, velocity = velocity_by_LK(V, t, x, y, timeafter, coarse=coarse)

    ax1 = plt.subplot(121)
    plt.quiver(gxB, gyB, gvxB, gvyB, color='blue')
    plt.quiver(gx, gy, gvx, gvy, color='red')
    set_axis_hidens(ax1)

    ax2 = plt.subplot(122)
    plt.hist(np.ravel(velocityB), bins=200, edgecolor='None', color='blue', alpha=0.5)
    plt.hist(np.ravel(velocity), bins=200, edgecolor='None', color='red', alpha=0.5)

    plt.show()


def velocity_by_LK(V, t, x, y, time, coarse=True):
    tspacing = np.abs((np.median(np.diff(t))))
    xspacing = np.abs((np.median(np.diff(x))))/2
    yspacing = np.abs((np.median(np.diff(y))))
    if coarse:
        xspacing, yspacing = 2 * xspacing, 2 * yspacing
    logging.info('Scale = (dx,dy,dt) = (%.3f um, %.3f um, %.3f ms)' % (xspacing, yspacing, tspacing))

    index = np.where(t == time)[0]
    grid_x, grid_y, grid_V1 = interpolate(x, y, np.ravel(V[:, index]), xspacing=xspacing, yspacing=yspacing)
    _, _, grid_V2 = interpolate(x, y, np.ravel(V[:, index + 1]), xspacing=xspacing, yspacing=yspacing)
    logging.info('Interpolation to %d points' % (np.shape(grid_x)[0] * np.shape(grid_x)[1]))
    xdot, ydot = LucasKanade(grid_V1, grid_V2)
    velocity = np.sqrt(xdot * xdot + ydot * ydot)
    return grid_x, grid_y, grid_V1, grid_V2, xdot, ydot, velocity


test_flows()