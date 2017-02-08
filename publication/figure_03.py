from hana.segmentation import find_AIS
from publication.plotting import FIGURE_NEURON_FILE, cross_hair, label_subplot, voltage_color_bar, adjust_position, without_spines_and_ticks
from hana.plotting import set_axis_hidens
from hana.recording import electrode_neighborhoods, load_traces
from hana.misc import ModelFunction

from scipy.stats import binom
import numpy as np
from matplotlib import pyplot as plt
import pylab as pl
from matplotlib import cm
from matplotlib import patches as ptc
from mpl_toolkits.axes_grid.inset_locator import inset_axes

from statsmodels import robust
import logging
logging.basicConfig(level=logging.DEBUG)



# Final figure 3

def figure3():
    # Load electrode coordinates
    radius = 200
    neighbors = electrode_neighborhoods(mea='hidens', neighborhood_radius=radius)

    # Load example data
    V, t, x, y, trigger, neuron = load_traces(FIGURE_NEURON_FILE)
    t *= 1000  # convert to ms

    # Position of the AIS
    index_AIS = find_AIS(V)
    x_AIS = x[index_AIS]
    y_AIS = y[index_AIS]

    # Select axonal AP
    center_id = 4961
    V = V[:, t==1]
    new_V, new_x, new_y = within_neighborhood(V, x, y, 4961, neighbors)
    r, theta = cart2pol(new_x, new_y)

    # Select good signals from electrodes in the area perpendicular to axon
    phi = np.pi * 5/6
    epsilon = np.pi / 5
    good = np.logical_or(np.logical_or(abs(theta-phi)<epsilon,abs(theta-phi+np.pi)<epsilon),r<10)
    not_good = np.logical_not(good)

    # Fit spatial attenuation
    A = (new_V+np.median(new_V[good]))/min(new_V[good])
    model = ModelFunction(formula_string='z / np.sqrt(z * z + x * x)',
                          bounds_dict=dict(z=[0,20]))
    model.fit(np.ravel(r[good]), np.ravel(A[good]))
    rfit = np.linspace(-radius, radius, 401)
    Afit = model.predict(rfit)

    # Summary
    print (model.parameters)


    # Plotting
    fig = plt.figure('Figure 3', figsize=(13, 7))
    fig.suptitle('Figure 3. Spatial spread of axonal signals', fontsize=14, fontweight='bold')

    ax1 = plt.subplot(231)
    img = plt.imread('data/hdmea_neighborhoods.png')
    plt.imshow(img)
    plt.axis('off')

    # Potential distribution around point source (axon)
    # Assuming the HDMEA surface to be an insulator and the extracellular space to be homogeneous and isotropic, the
    # potential distribution is the same as for two charges in free medium, one mirrored at the surface
    ax2 = plt.subplot(232)
    plot_potential(ax2)
    ax2.set_xlabel(r'radius, r [$\mu$m]')
    ax2.set_ylabel(r'distance , z [$\mu$m]')
    # without_spines_and_ticks(ax2)
    ax2.annotate('', (35, 0), (35, 10), arrowprops=dict(shrinkB=0, shrinkA=0, arrowstyle='<->'))
    plt.hlines(10, 0, 35, color='black', linestyle=':', zorder=10)
    ax2.text(35, 3, r' z', size=15)
    ax2.text(-45, 80, r'Potential $\Phi(d,r)$', color='gray', size=12,
             bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    adjust_position(ax2,xshrink=0.01,yshrink=0.01, yshift=0.01)

    # Potential at HDMEA surface
    ax3 = plt.subplot(233)
    for z in (3,30):
        thA = model.predict(rfit, override=dict(z=z))
        ax3.plot(rfit, thA, label=r'z=%1.1f $\mu$m' % z)
    ax3.legend(frameon=False,prop={'size':12})
    ax3.text(-45, 1.4, r'$A \approx \frac{z}{\sqrt{r^2+z^2}}$', size=14,
             bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    annotate_r_arrow(ax3, -18)
    annotate_r_arrow(ax3, +18)
    ax3.set_xlabel(r'radius, r [$\mu$m]')
    ax3.set_ylabel(r'$A = V/V_{n}$')
    ax3.set_xlim((-50, 50))
    ax3.set_ylim((-0.1, 1.6))
    ax3.plot((-radius,+radius),(0,0), 'k:')
    # without_spines_and_ticks(ax3)
    adjust_position(ax3,xshrink=0.01,yshrink=0.01, yshift=0.01)


    # Map voltage at a time
    ax4 = plt.subplot(234)
    ax4_h1 = ax4.scatter(x, y, c=V, s=20, marker='o',
                         edgecolor='None', cmap='seismic')
    voltage_color_bar(ax4_h1, label=r'$V$ [$\mu$V]')
    cross_hair(ax4, x_AIS, y_AIS, color='black')
    neighborhood =plt.Circle((x[center_id], y[center_id]), radius, edgecolor='green', facecolor='None')
    ax4.add_artist(neighborhood)
    set_axis_hidens(ax4)

    ax5 = plt.subplot(235, polar=True)
    ax5_h1 = plt.scatter(theta, r, c=new_V, s=50, marker='o', edgecolor='None', cmap='seismic')
    plot_pie(ax5, phi, epsilon)
    ax5.set_ylim(0,radius)
    ax5_h1.set_clim(vmin=-20, vmax=20)
    adjust_position(ax5, xshift=-0.01)
    adjust_position(ax5, yshift=-0.02)
    ax5.set_yticklabels(('',))
    ax5.set_xticklabels(('',))

    ax6 = plt.subplot(236)
    ax6.scatter(-r[good], A[good], c='green', s=20, marker='o', edgecolor='None')
    ax6.scatter(r[good], A[good], c='green', s=20, marker='o', edgecolor='None', label='fitted data')
    ax6.scatter(r[not_good], A[not_good], c='gray', s=20, marker='o', edgecolor='None', label='excluded', zorder=0)
    ax6.scatter(-r[not_good], A[not_good], c='gray', s=20, marker='o', edgecolor='None', zorder=0)
    ax6.plot(rfit,Afit, label=r'fit, z=%1.1f $\mu$m' % model.parameters['z'])
    ax6.plot((-radius,+radius),(0,0), 'k:')
    ax6.set_xlim(-radius,radius)
    ax6.legend(loc=2, frameon=False,prop={'size':12}, scatterpoints=1)
    ax6.set_ylim(-0.1,1.6)
    ax6.set_ylabel(r'$A = V/V_{n}$')
    ax6.set_xlabel (r'radius [$\mu$m]')
    ax6.xaxis.set_ticks(np.linspace(-200, 200, 5))

    # ax3sub = inset_axes(ax3,
    #                         width="25%",
    #                         height="25%",
    #                         loc=1)
    # ax3sub.plot(rfit,Afit)
    # ax3sub.set_ylim(0,1)
    # ax3sub.set_xlim(-30,30)

    label_subplot(ax1, 'A', xoffset=-0.05, yoffset=-0.01)
    label_subplot(ax2, 'B', xoffset=-0.05, yoffset=-0.01)
    label_subplot(ax3, 'C', xoffset=-0.05, yoffset=-0.01)
    label_subplot(ax4, 'D', xoffset=-0.05, yoffset=-0.01)
    label_subplot(ax5, 'E', xoffset=-0.00, yoffset=0.00)
    label_subplot(ax6, 'F', xoffset=-0.05, yoffset=-0.01)

    plt.show()


def plot_potential(ax3, z0=10):
    r_range = np.linspace(-50, 50, 101)
    z_range = np.linspace(0, 90, 90)
    r, z = np.meshgrid(r_range, z_range)
    z0_mirror = -z0
    phi = 1.0 / np.sqrt(np.power(r, 2) + np.power(z - z0, 2)) + 1.0 / np.sqrt(
        np.power(r, 2) + np.power(z - z0_mirror, 2))
    ax3.contour(r, z, phi, levels=np.linspace(0, 0.6, 50), colors='gray', linestyles='solid')
    plt.plot(0, z0, 'r.', markersize=10)
    plt.annotate('point source', (0, z0), (10, 50), size=12, arrowprops={'arrowstyle': '->', 'color': 'black'})


def annotate_r_arrow(ax4, radius):
    ax4.annotate('', (0, 0.05), (radius, 0.05), arrowprops=dict(shrinkB=0, shrinkA=0, arrowstyle='<->'))
    ax4.text(radius/2, 0.07, r'r', size=15)


def plot_pie(ax1, phi, epsilon):
    r = (0, 250, 250, 0, 250, 250, 0)
    theta = (0, phi - epsilon, phi + epsilon, 0, np.pi + phi - epsilon, np.pi + phi + epsilon, 0)
    polygon = ptc.Polygon(zip(theta, r), color='green', alpha=0.2)
    ax1.add_line(polygon)


def plot_polar_neigborhood(V, ax, x, y, center_id, neighbors):
    new_V, new_x, new_y = within_neighborhood(V, x, y, center_id, neighbors)
    r, theta = cart2pol(new_x, new_y)
    ax = replace_axis(ax, polar=True)
    c = plt.scatter(theta, r, c=new_V, s=50, marker='o', edgecolor='None', cmap='seismic')
    ax.set_ylim(0,200)
    voltage_color_bar(c, label=r'$V$ [$\mu$V]')
    return c


def cart2pol(x, y):
    z = x - 1j * y
    theta = np.angle(z)
    r = np.abs(z)
    return r, theta


def within_neighborhood(V, x, y, center_id, neighbors):
    center_x, center_y = x[center_id], y[center_id]
    neighborhood = neighbors[center_id]
    neighborhood_V = V[neighborhood]
    neighborhood_x, neighborhood_y = x[neighborhood] - center_x, y[neighborhood] - center_y
    return neighborhood_V, neighborhood_x, neighborhood_y


def replace_axis(ax, **kwargs):
    position = ax.get_position()
    plt.delaxes(ax)
    ax = plt.axes(position, **kwargs)
    return ax

figure3()
# spread()
