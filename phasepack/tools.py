import numpy as np
from scipy.fftpack import fftshift, ifftshift

# Try and use the faster Fourier transform functions from the pyfftw module if
# available
try:
    from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
# Otherwise use the normal scipy fftpack ones instead (~2-3x slower!)
except ImportError:
    import warnings
    warnings.warn("""
Module 'pyfftw' (FFTW Python bindings) could not be imported. To install it, try
running 'pip install pyfftw' from the terminal. Falling back on the slower
'fftpack' module for 2D Fourier transforms.""")
    from scipy.fftpack import fft2, ifft2


def lowpassfilter(size, cutoff, n):
    """
    Constructs a low-pass Butterworth filter:

        f = 1 / (1 + (w/cutoff)^2n)

    usage:  f = lowpassfilter(sze, cutoff, n)

    where:  size    is a tuple specifying the size of filter to construct
            [rows cols].
        cutoff  is the cutoff frequency of the filter 0 - 0.5
        n   is the order of the filter, the higher n is the sharper
            the transition is. (n must be an integer >= 1). Note
            that n is doubled so that it is always an even integer.

    The frequency origin of the returned filter is at the corners.
    """

    if cutoff < 0. or cutoff > 0.5:
        raise Exception('cutoff must be between 0 and 0.5')
    elif n % 1:
        raise Exception('n must be an integer >= 1')
    if len(size) == 1:
        rows = cols = size
    else:
        rows, cols = size

    if (cols % 2):
        xvals = np.arange(-(cols - 1) / 2.,
                          ((cols - 1) / 2.) + 1) / float(cols - 1)
    else:
        xvals = np.arange(-cols / 2., cols / 2.) / float(cols)

    if (rows % 2):
        yvals = np.arange(-(rows - 1) / 2.,
                          ((rows - 1) / 2.) + 1) / float(rows - 1)
    else:
        yvals = np.arange(-rows / 2., rows / 2.) / float(rows)

    x, y = np.meshgrid(xvals, yvals, sparse=True)
    radius = np.sqrt(x * x + y * y)

    return ifftshift(1. / (1. + (radius / cutoff) ** (2. * n)))


def rayleighmode(data, nbins=50):
    """
    Computes mode of a vector/matrix of data that is assumed to come from a
    Rayleigh distribution.

    usage:  rmode = rayleighmode(data, nbins)

    where:  data    data assumed to come from a Rayleigh distribution
            nbins   optional number of bins to use when forming histogram
                    of the data to determine the mode.

    Mode is computed by forming a histogram of the data over 50 bins and then
    finding the maximum value in the histogram. Mean and standard deviation
    can then be calculated from the mode as they are related by fixed
    constants.

        mean = mode * sqrt(pi/2)
        std dev = mode * sqrt((4-pi)/2)

    See:
        <http://mathworld.wolfram.com/RayleighDistribution.html>
        <http://en.wikipedia.org/wiki/Rayleigh_distribution>
    """
    n, edges = np.histogram(data, nbins)
    ind = np.argmax(n)
    return (edges[ind] + edges[ind + 1]) / 2.


def perfft2(im, compute_P=True, compute_spatial=False):
    """
    Moisan's Periodic plus Smooth Image Decomposition. The image is
    decomposed into two parts:

        im = s + p

    where 's' is the 'smooth' component with mean 0, and 'p' is the 'periodic'
    component which has no sharp discontinuities when one moves cyclically
    across the image boundaries.

    useage: S, [P, s, p] = perfft2(im)

    where:  im      is the image
            S       is the FFT of the smooth component
            P       is the FFT of the periodic component, returned if
                    compute_P (default)
            s & p   are the smooth and periodic components in the spatial
                    domain, returned if compute_spatial

    By default this function returns `P` and `S`, the FFTs of the periodic and
    smooth components respectively. If `compute_spatial=True`, the spatial
    domain components 'p' and 's' are also computed.

    This code is adapted from Lionel Moisan's Scilab function 'perdecomp.sci'
    "Periodic plus Smooth Image Decomposition" 07/2012 available at:

        <http://www.mi.parisdescartes.fr/~moisan/p+s>
    """

    if im.dtype not in ['float32', 'float64']:
        im = np.float64(im)

    rows, cols = im.shape

    # Compute the boundary image which is equal to the image discontinuity
    # values across the boundaries at the edges and is 0 elsewhere
    s = np.zeros_like(im)
    s[0, :] = im[0, :] - im[-1, :]
    s[-1, :] = -s[0, :]
    s[:, 0] = s[:, 0] + im[:, 0] - im[:, -1]
    s[:, -1] = s[:, -1] - im[:, 0] + im[:, -1]

    # Generate grid upon which to compute the filter for the boundary image
    # in the frequency domain.  Note that cos is cyclic hence the grid
    # values can range from 0 .. 2*pi rather than 0 .. pi and then pi .. 0
    x, y = (2 * np.pi * np.arange(0, v) / float(v) for v in (cols, rows))
    cx, cy = np.meshgrid(x, y)

    denom = (2. * (2. - np.cos(cx) - np.cos(cy)))
    denom[0, 0] = 1.     # avoid / 0

    S = fft2(s) / denom
    S[0, 0] = 0      # enforce zero mean

    if compute_P or compute_spatial:

        P = fft2(im) - S

        if compute_spatial:
            s = ifft2(S).real
            p = im - s

            return S, P, s, p
        else:
            return S, P
    else:
        return S
