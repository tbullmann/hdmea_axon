import logging

import numpy as np
from matplotlib import pyplot as plt

from hana.plotting import set_axis_hidens
from hana.recording import load_traces
from publication.plotting import FIGURE_NEURON_FILE_FORMAT

logging.basicConfig(level=logging.DEBUG)


class AffineTransformation:
    """
    T_AB dot A = B

    T_AB = A dot B^-1
    T_AB = T_AB^-1

    B = T_AB dot A
    A = T_BA dot B
    """

    def __init__(self, coordinates_A, coordinates_B):
        self.match(coordinates_A, coordinates_B)

    def match(self, coordinates_A, coordinates_B):
        ones = np.ones((1, np.shape(coordinates_A)[1]))
        B = np.vstack((coordinates_B, ones))
        A = np.vstack((coordinates_A, ones))
        self.T_AB = np.dot(A, np.linalg.inv(B))
        self.T_BA = np.linalg.inv(self.T_AB)

    def forward(self, coordinates_A):
        coordinates_B = self.transform(coordinates_A, self.T_BA)
        return coordinates_B

    def backward(self, coordinates_B):
        coordinates_A = self.transform(coordinates_B, self.T_AB)
        return coordinates_A

    @staticmethod
    def transform(coordinates_A, T):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        if coordinates_A.ndim==2:
            n_coordinates = np.shape(coordinates_A)[1]
        else:  # there was only one point, make its coordinates a 2 dimensional row vector
            n_coordinates = 1
            coordinates_A = np.atleast_2d(coordinates_A).T

        ones = np.ones((1, n_coordinates))
        A = np.vstack((coordinates_A, ones))
        B = np.dot(T, A)
        coordinates_B = B[:2, :]
        return coordinates_B


class HidensTransformation(AffineTransformation):
    """
    Transformation
    """
    def __init__(self, x, y, origin='zero'):

        # Not really important choice, because later shifted to yield (xmin, ymin) <-> (0,0)
        if origin=='min':  # works fine
            (xmin, ymin) = (np.amin(x), np.amin(y))
        if origin == 'first':   # works fine too
            (xmin, ymin) = (np.asscalar(x[0]),np.asscalar(y[0]))
        if origin=='zero':  # works fine as well, make it default
            xmin, ymin = 0, 0

        # Assuming that j axis aligns with x axis and y axis is perpendicular to x
        xspacing = np.median(np.unique(np.diff(np.sort(x))))
        yspacing = np.median(np.unique(np.diff(np.sort(y))))

        logging.info('Initial transformation')
        cart = np.array([[xmin,  xmin-xspacing, xmin+xspacing],
                         [ymin,  ymin+yspacing, ymin+yspacing]])
        hex = np.array( [[0,          1,         0],
                         [0,          0,         1]])
        logging.info(cart)
        AffineTransformation.__init__(self, cart, hex)

        # Backtransforming the origin of the hexagonal grid to cartesian coordinates
        i, j = self.xy2ij(x,y, type=float)  # Need float instead of integers, because origin not yet at (0,0)
        x0, y0 = self.ij2xy(np.min(i),np.min(j))
        xmin = np.asscalar(x0)
        ymin = np.asscalar(y0)
        logging.info('Final transformation')  # shifting (xmin, ymin) to yield (0,0)
        cart = np.array([[xmin, xmin - xspacing, xmin + xspacing],
                         [ymin, ymin + yspacing, ymin + yspacing]])
        logging.info(cart)
        self.match(cart, hex)

        # Store x and y for use in subset
        self.x = x
        self.y = y

    def xy2ij (self, x, y, type=int):
        hex = self.forward(np.array([x, y]))
        hex = np.round(hex).astype(type) if type==int else hex
        i = hex[0, :]
        j = hex[1, :]
        return i, j

    def ij2xy(self, i,j):
        cart = self.backward(np.array([i, j]).astype(float))
        x = cart[0, :]
        y = cart[1, :]
        return x, y

    def subset(self, V, period=2):
        """
        Subsets the hidens elecectrodes to yield period time larger hexagons (see test_subset)
        :param V: some value, usually a voltage V(x,y,t) = V[el_idx, t]
        :param period: size of hexagons in multiples of the original hexagons
        :return: x, y, V: subset of x, y, V
        """
        i, j = self.xy2ij(self.x, self.y)
        index = i % period * j % period == 1
        x, y, V = self.x[index], self.y[index], V[index, :]
        return x, y, V


def test_subset(neuron=5):
    filename = FIGURE_NEURON_FILE_FORMAT % neuron
    V, t, x, y, trigger, neuron = load_traces(filename)

    hidens = HidensTransformation(x, y)
    xs, ys, Vs = hidens.subset(V)

    ax = plt.subplot(111)
    plt.plot(x, y, 'o', markerfacecolor='None', markeredgecolor='black', label='all')
    plt.plot(xs, ys, 'ko', label='subset')
    set_axis_hidens(ax)
    plt.legend(numpoints=1)

    plt.show()


def test_grid(neuron=5):
    filename = FIGURE_NEURON_FILE_FORMAT % neuron
    V, t, x, y, trigger, neuron = load_traces(filename)

    hidens = HidensTransformation(x, y)

    i, j = hidens.xy2ij(x,y)
    xb, yb = hidens.ij2xy(i,j)
    x0, y0 = hidens.ij2xy(0,0)

    ax1 = plt.subplot(121)
    plt.plot(x, y, 'bx', label='original')
    plt.plot(xb, yb, 'k+', label='backtransformed')
    plt.plot(x0, y0, 'go', label='hexagonal origin (backtransformed)')
    plt.title ('cartesian coordinates x,y')
    set_axis_hidens(ax1)
    ax1.set_ylim((-500,2200))
    plt.legend(numpoints=1)

    ax2 = plt.subplot(122)
    plt.plot(i, j, 'ko', label='transformed')
    plt.plot(0, 0, 'go', label='hexagonal origin')
    plt.title('hexagonal grid index i, j')
    ax2.set_xlim((-1,np.amax(i)))
    ax2.set_ylim((-1, np.amax(j)))
    plt.legend(numpoints=1)

    plt.show()


if __name__ == "__main__":
    test_grid()
    test_subset()