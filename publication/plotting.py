import numpy as np
from matplotlib import pyplot as plt


FIGURE_ARBORS_FILE = 'data/hidens2018at35C_arbors.mat'
FIGURE_EVENTS_FILE = 'data/hidens2018at35C_events.mat'
FIGURE_ELECTRODES_FILE = 'data/hidens_electrodes.mat'

def plot_parameter_dependency(ax, Z, x, y, w=None, levels=None, fmt='%d'):
    """Plotting parameter dependency"""
    from scipy.ndimage.filters import gaussian_filter
    import matplotlib.patches as mpatches
    X, Y = np.meshgrid(x, y)
    if w is None:  # standard black and white blot
        CS1 = ax.contour(X, Y, gaussian_filter(Z, 1), levels=levels, colors='k')
        ax.clabel(CS1, fmt=fmt, inline=1, fontsize=10, inline_spacing=-5)
    else:  # first entry entry in dictionary in black, second in red and dashed lines
        CS1 = ax.contour(X, Y, gaussian_filter(Z[w[0]], 1), levels=levels, colors='k')
        ax.clabel(CS1, fmt=fmt, inline=1, fontsize=10, inline_spacing=-5)
        CS2 = ax.contour(X, Y, gaussian_filter(Z[w[1]], 1), levels=levels, colors='r',
                         linestyles='dashed')
        ax.clabel(CS2, fmt=fmt, inline=1, fontsize=10, inline_spacing=-5)
        # Zinf = np.float64(np.isfinite(Z[w[0]]))
        # CS1 = ax.contourf(X, Y, gaussian_filter(Zinf, 1), levels=(0,1), colors='y')
        black_patch = mpatches.Patch(color='black', label=w[0])
        red_patch = mpatches.Patch(color='red', label=w[1])
        plt.legend(loc=2, handles=[black_patch, red_patch])


