import logging

import numpy as np
from matplotlib import pyplot as plt

from future_publication.flow import LucasKanade, interpolate
from hana.plotting import set_axis_hidens
from hana.recording import load_traces
from publication.plotting import FIGURE_NEURON_FILE_FORMAT

logging.basicConfig(level=logging.DEBUG)


class affine_transform:
    """
    T_AB dot A = B
    """

    def __init__(self, coordinates_A, coordinates_B):
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
        ones = np.ones((1, np.shape(coordinates_A)[1]))
        A = np.vstack((coordinates_A, ones))
        B = np.dot(T, A)
        coordinates_B = B[:2, :]
        return coordinates_B


class transform_hidens_grid(affine_transform):
    def __init__(self, x, y, zero_at_min=False):
        (xmin, ymin) = (np.amin(x), np.amin(y)) if zero_at_min else (x[0],y[0])
        xspacing = np.median(np.unique(np.diff(np.sort(x))))
        yspacing = np.median(np.unique(np.diff(np.sort(y))))
        cart = np.array([[xmin,  xmin-xspacing, xmin+xspacing],
                         [ymin,  ymin+yspacing, ymin+yspacing]])
        hex = np.array( [[0,          1,         0],
                         [0,          0,         1]])
        affine_transform.__init__(self, cart, hex)

    def xy2ij (self, x, y):
        hex = self.forward(np.array([x, y]))
        i = np.round(hex[0, :]).astype(int)
        j = np.round(hex[1, :]).astype(int)
        return i, j

    def ij2xy(self, i,j):
        cart = self.backward(np.array([i, j]).astype(float))
        x = cart[0, :]
        y = cart[1, :]
        return x, y


def test_grid(neuron=5, coarse=True):
    filename = FIGURE_NEURON_FILE_FORMAT % neuron
    V, t, x, y, trigger, neuron = load_traces(filename)

    xspacing = np.mean(np.diff(np.unique(np.sort(x))))
    yspacing = np.mean(np.diff(np.unique(np.sort(y))))

    cart = np.array([[0, -xspacing, +xspacing],
                     [0, yspacing, yspacing]])

    hex = np.array([[0, 1, 0],
                    [0, 0, 1]])






    cart2hex = transform_hidens_grid(x,y)


    print cart2hex.forward(cart)

    i, j = cart2hex.xy2ij(x, y)

    x, y = cart2hex.ij2xy(i, j)

    plt.subplot(121)
    plt.plot(x,y,'ko')


    plt.subplot(122)
    plt.plot(i, j, 'ko')

    plt.show()

test_grid()