import logging

import numpy as np
from scipy.interpolate import griddata
from scipy.signal import convolve2d


def filter2(im, kernel):
    return convolve2d(im, kernel, 'same')


def computeDerivatives(im1, im2):
    """
    https://github.com/scienceopen/Optical-Flow-LucasKanade-HornSchunck/blob/master/HornSchunck.py
    :param im1, im2: consecutive images
    :return: derivatives for x, y and t axis (in pixels)
    """
    # build kernels for calculating derivatives
    kernelX = np.array([[-1, 1],
                         [-1, 1]]) * .25  # kernel for computing d/dx
    kernelY = np.array([[-1,-1],
                         [ 1, 1]]) * .25  # kernel for computing d/dy
    kernelT = np.ones((2,2))*.25

    fx = filter2(im1, kernelX) + filter2(im2, kernelX)
    fy = filter2(im1, kernelY) + filter2(im2, kernelY)

    # ft = im2 - im1
    ft = filter2(im1, kernelT) + filter2(im2, -kernelT)

    return fx, fy, ft


def HS(im1, im2, alpha=1, Niter=10, scale=(1,1,1)):
    """
    Source: https://github.com/scienceopen/Optical-Flow-LucasKanade-HornSchunck/blob/master/HornSchunck.py
    Note: Very fast, but sensitive to noise, hence no convergent pixel take over.
    :param im1, im2: consecutive images
    :param alpha: smoothness constraint
    :param Niter: iterations
    :param scale: scaling for x, y and t axis, default scale = (dx,dy,dt) = (1,1,1)
    :return: u, v: dense velocity in x and y direction
    """

    #set up initial velocities
    uInitial = np.zeros_like(im1)
    vInitial = np.zeros_like(im1)

    # Set initial value for the flow vectors
    u = uInitial
    v = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    # Averaging kernel
    kernel=np.array([[1/12, 1/6, 1/12],
                     [1/6,    0, 1/6],
                     [1/12, 1/6, 1/12]],float)

    # Iteration to reduce error
    for iteration in range(Niter):

        logging.info('Iteration %d' % iteration)

        # Compute local averages of the flow vectors
        uAvg = filter2(u,kernel)
        vAvg = filter2(v,kernel)

        # Common part of update step
        der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)

        # iterative step
        u = uAvg - fx * der
        v = vAvg - fy * der

    # Set diverging pixels to zero
    valid = np.logical_and(np.isfinite(u), np.isfinite(v))
    u[np.logical_not(valid)] = 0
    v[np.logical_not(valid)] = 0

    # Normalize
    dx, dy, dt = scale
    u = u * dx/dt
    v = v * dy/dt

    return u, v


def LucasKanade(im1, im2, w=2, scale=(1, 1, 1)):
    """
    Source: Translated from matlab this matlab code piece:
    http://de.mathworks.com/matlabcentral/fileexchange/48744-lucas-kanade-tutorial-example-1/content/LucasKanadeExample1/html/LKExample1.html
    :param im1, im2: consecutive images
    :param scale: scaling for x, y and t axis, default scale = (dx,dy,dt) = (1,1,1)
    :return: u, v: dense velocity in x and y direction
    """

    fx, fy, ft = computeDerivatives(im1, im2)

    u = np.zeros_like(im1)
    v = np.zeros_like(im2)

    for i in range(w, np.shape(fx)[0] - w):

        for j in range(w, np.shape(fx)[1]-w):

            Ix = fx[i-w:i+w+1, j-w:j+w+1]
            Iy = fy[i-w:i+w+1, j-w:j+w+1]
            It = ft[i-w:i+w+1, j-w:j+w+1]

            Ix = np.ravel(Ix)
            Iy = np.ravel(Iy)
            b = -np.ravel(It)  # get b here

            A = np.vstack((Ix, Iy))  # get A here
            nu = np.dot(np.linalg.pinv(A).T,b)  # get velocity here

            u[i, j] = nu[0]
            v[i, j] = nu[1]

    # Normalize
    dx, dy, dt = scale
    u = u * dx / dt
    v = v * dy / dt

    return u, v


def interpolate(x, y, z, xspacing = 20, yspacing=20):

    grid_x, grid_y = np.meshgrid(np.arange(min(x),max(x),xspacing),
                                 np.arange(min(y),max(y),yspacing),
                                 sparse=False,
                                 indexing='ij')
    grid_z2 = griddata(np.vstack((x,y)).T,
                       np.ravel(z),
                       (grid_x, grid_y),
                       method='cubic')

    return grid_x, grid_y, grid_z2