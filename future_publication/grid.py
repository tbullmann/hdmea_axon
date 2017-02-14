import logging

import numpy as np
from matplotlib import pyplot as plt

from future_publication.flow import LucasKanade, interpolate
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
    def __init__(self, x, y, zero_at_min=True):
        if zero_at_min:
            (xmin, ymin) = (np.amin(x), np.amin(y))
        else:
            (xmin, ymin) = (np.asscalar(x[0]),np.asscalar(y[0]))
        xmin, ymin = 0, 0

        xspacing = np.median(np.unique(np.diff(np.sort(x))))
        yspacing = np.median(np.unique(np.diff(np.sort(y))))
        cart = np.array([[xmin,  xmin-xspacing, xmin+xspacing],
                         [ymin,  ymin+yspacing, ymin+yspacing]])
        hex = np.array( [[0,          1,         0],
                         [0,          0,         1]])
        logging.info('Initial transformation')
        logging.info(cart)
        AffineTransformation.__init__(self, cart, hex)
        i, j = self.xy2ij(x,y, type=float)
        x0, y0 = self.ij2xy(np.min(i),np.min(j))
        xmin=np.asscalar(x0)
        ymin=np.asscalar(y0)
        cart = np.array([[xmin, xmin - xspacing, xmin + xspacing],
                         [ymin, ymin + yspacing, ymin + yspacing]])
        logging.info('Final transformation')
        logging.info(cart)
        self.match(cart, hex)

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


def test_grid(neuron=5, coarse=True):
    filename = FIGURE_NEURON_FILE_FORMAT % neuron
    V, t, x, y, trigger, neuron = load_traces(filename)

    hidens = HidensTransformation(x, y)

    i, j = hidens.xy2ij(x, y)

    x2, y2 = hidens.ij2xy(i, j)

    ax1 = plt.subplot(121)
    plt.plot(x,y,'b+')
    plt.plot(x2,y2,'kx')
    x0, y0 = hidens.ij2xy(np.min(i), np.min(j))
    plt.plot(x0,y0,'ro')
    # set_axis_hidens(ax1)


    ax2 = plt.subplot(122)
    plt.plot(i, j, 'ko')
    ax2.axis('equal')

    plt.show()

test_grid()