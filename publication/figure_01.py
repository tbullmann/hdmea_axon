import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
from hana.recording import load_traces
from hana.plotting import set_axis_hidens
from publication.plotting import FIGURE_NEURON_FILE

# Load example data
V, t, x, y, trigger, neuron = load_traces(FIGURE_NEURON_FILE)
t *= 1000  # convert to ms
mpl.rcParams['contour.negative_linestyle'] = 'solid'

indices = list(*np.where(np.logical_and(t>=0, t<2)))
color=iter(cm.rainbow(np.linspace(0,1,len(indices))))

for i in indices:

    c = next(color)

    if i%4==0:
        plt.vlines(-10,-10,-10,colors=c, label = '%1.1fms' % t[i])  # proxy for contour line legend

    frame = V[:,i]

    new_x_range = np.arange(min(x), max(x), 5)
    new_y_range = np.arange(min(y), max(y), 5)
    new_x, new_y = np.meshgrid(new_x_range, new_y_range, indexing='ij')

    Z = griddata(zip(x,y), frame, (new_x, new_y), method='cubic')

    # CS1 = plt.contourf(new_x, new_y, Z, colors='gray', levels=(-600, -5),zorder=2*i)
    CS2 = plt.contour(new_x, new_y, Z, colors=(c,), levels=(-5,))

set_axis_hidens(ax1)
plt.legend(frameon=False)
plt.show()



