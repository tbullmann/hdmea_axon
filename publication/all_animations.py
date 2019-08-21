import os
import numpy as np
from scipy.interpolate import griddata
import matplotlib.animation as animation
from matplotlib import pyplot as plt

from hana.plotting import mea_axes
from publication.data import FIGURE_CULTURE, FIGURE_NEURONS
from publication.experiment import Experiment
from publication.plotting import voltage_color_bar


def make_movies(moviepath=None, dpi=72):

    culture = FIGURE_CULTURE

    for neuron in FIGURE_NEURONS:

        # Make movie from spike-triggered averages
        V, t, x, y, trigger, neuron = Experiment(culture).traces(neuron)
        ani = make_movie(V, t, x, y, culture, neuron)

        # Saving the animation
        save_movie(ani, dpi, moviepath, 'neuron%d' % neuron)


def save_movie(ani, dpi, moviepath, basename):
    """
    Saves an animation as gif or mp4 using imagemagick or ffmpeg, respectively
    :param ani: animation object
    :param dpi: dot per inch for output image scaling
    :param moviepath: path where movies are stored
    :param basename: basename of that particular animation file
    :return:
    """
    filename = os.path.join(moviepath, basename)
    saved = False
    if 'imagemagick' in animation.writers.avail:
        writer = animation.writers['imagemagick'](fps=30)
        ani.save(filename + '.gif', writer=writer, dpi=dpi)
        saved = True
    if 'ffmpeg' in animation.writers.avail:
        writer = animation.writers['imagemagick'](fps=30)
        ani.save(filename + '.mp4', writer=writer, dpi=dpi)
        saved = True
    if saved:
        print ('Saved ' + filename)
    else:
        print ('Please install video codec, e.g. imagemagick or ffmpeg')


def make_movie(V, t, x, y, culture, neuron):
    """

    :param V: spike triggered averages (traces) for each electrode
    :param t: time for each frame (relative to trigger, in ms)
    :param x, y: coordinates for each electrode
    :param culture, neuron: for title
    :return: ani: animation handle
    """

    t_ms = t * 1000  # ms
    n_frames = len(t)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = ax.imshow(interpolate_frame(V[:, 0], x, y),
                   extent=[min(x), max(x), max(y), min(y)],
                   cmap='seismic',
                   interpolation='nearest')
    ax.set_title('Culture %d, Neuron %d, Time %1.1f' % (culture, neuron, t_ms[0]))
    voltage_color_bar(im, label=r'$V$ [$\mu$V]', shrink=0.5)
    mea_axes(ax)
    fig.set_size_inches([5, 5])
    plt.tight_layout()

    def update_frame(n):
        tmp = interpolate_frame(V[:, n], x, y)
        im.set_data(tmp)
        ax.set_title('Culture %d, Neuron %d, %1.3f ms' % (culture, neuron, t_ms[n]))
        return im

    ani = animation.FuncAnimation(fig, update_frame, n_frames, interval=30)

    plt.cla
    return ani


def interpolate_frame(frame, x, y, resolution=5):
    """
    Interpolate frames to grid.
    :param frame: spike triggered averages frame for each electrode
    :param x, y: coordinates for each electrode
    :param resolution: grid spacing in um
    :return: Z interpolated frame (as an image)
    """
    new_x_range = np.arange(min(x), max(x), resolution)
    new_y_range = np.arange(min(y), max(y), resolution)
    new_x, new_y = np.meshgrid(new_x_range, new_y_range, indexing='ij')
    Z = griddata(zip(x, y), frame, (new_x, new_y), method='linear')
    return Z


if __name__ == "__main__":
    make_movies(Experiment(FIGURE_CULTURE).results_directory)