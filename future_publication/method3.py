import logging

import numpy as np
from matplotlib import pyplot as plt

from future_publication.flow import LucasKanade, interpolate
from hana.plotting import set_axis_hidens
from hana.recording import load_traces
from publication.plotting import FIGURE_NEURON_FILE_FORMAT

logging.basicConfig(level=logging.DEBUG)


def test_flows(neuron=5):
    filename = FIGURE_NEURON_FILE_FORMAT % neuron
    V, t, x, y, trigger, neuron = load_traces(filename)

    t *= 1000  # convert to ms

    xspacing = np.abs((np.median(np.diff(x))))
    yspacing = np.abs((np.median(np.diff(y))))*2

    grid_x, grid_y, grid_V1 = interpolate(x, y, np.ravel(V[:, t == 1]), xspacing=xspacing, yspacing=yspacing)
    _,_,grid_V2 = interpolate(x, y, np.ravel(V[:, t == 1 + 0.05]), xspacing=xspacing, yspacing=yspacing)


    ax = plt.subplot(111)
    CS2 = plt.contour(grid_x, grid_y, -grid_V1, colors=('red',), levels=(5,))
    CS2 = plt.contour(grid_x, grid_y, -grid_V2, colors=('green',), levels=(5,))

    u, v = LucasKanade(grid_V1, grid_V2)
    # u, v = HS(grid_V1,grid_V2)

    plt.quiver(grid_x,grid_y, u, v)

    set_axis_hidens(ax)

    plt.show()


test_flows()