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
    time = -4  # ms

    tspacing = np.abs((np.median(np.diff(t))))
    xspacing = np.abs((np.median(np.diff(x))))/2
    yspacing = np.abs((np.median(np.diff(y))))
    if coarse:
        xspacing, yspacing = 2 * xspacing, 2 * yspacing

    logging.info('Scale = (dx,dy,dt) = (%.3f um, %.3f um, %.3f ms)' %  (xspacing, yspacing, tspacing) )

    grid_x, grid_y, grid_V1 = interpolate(x, y, np.ravel(V[:, t==time]), xspacing=xspacing, yspacing=yspacing)
    _,_,grid_V2 = interpolate(x, y, np.ravel(V[:, t==time+0.05]), xspacing=xspacing, yspacing=yspacing)

    logging.info('Interpolation to %d points' % (np.shape(grid_x)[0]*np.shape(grid_x)[1]))

    xdot, ydot = LucasKanade(grid_V1, grid_V2)
    # xdot, ydot = HS(grid_V1,grid_V2)

    velocity = np.sqrt(xdot*xdot + ydot*ydot)

    ax1 = plt.subplot(121)
    plt.contour(grid_x, grid_y, -grid_V1, colors=('red',), levels=(5,))
    plt.contour(grid_x, grid_y, -grid_V2, colors=('green',), levels=(5,))
    plt.quiver(grid_x,grid_y, xdot, ydot)
    set_axis_hidens(ax1)

    ax2 = plt.subplot(222)
    plt.hist(np.ravel(velocity), bins=200)

    plt.show()


test_flows()