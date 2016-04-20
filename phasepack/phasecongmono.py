# MIT License:

# Permission is hereby  granted, free of charge, to any  person obtaining a
# copy of this software and associated  documentation files (the "Software"),
# to deal in the Software without restriction, subject to the following
# conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# The software is provided "as is", without warranty of any kind.

# Original MATLAB version by Peter Kovesi
# <http://www.csse.uwa.edu.au/~pk/research/matlabfns/PhaseCongruency/phasecongmono.m>

# Python translation by Alistair Muldal
# <alistair muldal@pharm ox ac uk>


import numpy as np
from scipy.fftpack import fftshift, ifftshift

from .tools import rayleighmode as _rayleighmode
from .tools import lowpassfilter as _lowpassfilter
from .tools import perfft2
from .filtergrid import filtergrid

# Try and use the faster Fourier transform functions from the pyfftw module if
# available
from .tools import fft2, ifft2


def phasecongmono(img, nscale=5, minWaveLength=3, mult=2.1, sigmaOnf=0.55,
                  k=2., cutOff=0.5, g=10., noiseMethod=-1, deviationGain=1.5):
    """
    Function for computing phase congruency on an image. This version uses
    monogenic filters for greater speed.

    Arguments:
    ------------------------
    <Name>      <Default>   <Description>
    img             N/A     The input image
    nscale          5       Number of wavelet scales, try values 3-6
    minWaveLength   3       Wavelength of smallest scale filter.
    mult            2.1     Scaling factor between successive filters.
    sigmaOnf        0.55    Ratio of the standard deviation of the Gaussian
                            describing the log Gabor filter's transfer function
                            in the frequency domain to the filter center
                            frequency
    k               2.0     No. of standard deviations of the noise energy
                            beyond the mean at which we set the noise threshold
                            point. You may want to vary this up to a value of
                            10 or 20 for noisy images
    cutOff          0.5     The fractional measure of frequency spread below
                            which phase congruency values get penalized
    g               10      Controls the 'sharpness' of the transition in the
                            sigmoid function used to weight phase congruency
                            for frequency spread
    noiseMethod     -1      Parameter specifies method used to determine
                            noise statistics.
                            -1 use median of smallest scale filter responses
                            -2 use mode of smallest scale filter responses
                            >=0 use this value as the fixed noise threshold
    deviationGain   1.5     Amplification to apply to the phase deviation
                            result. Increasing this sharpens the edge respones,
                            but can also attenuate their magnitude if the gain
                            is too large. Sensible values lie in the range 1-2.

    Return values:
    ------------------------
    M       Maximum moment of phase congruency covariance, which can be used
            as a measure of edge strength
    ori     Orientation image, in integer degrees (0-180), positive angles
            anti-clockwise.
    ft      Local weighted mean phase angle at every point in the image. A
            value of pi/2 corresponds to a bright line, 0 to a step and -pi/2
            to a dark line.
    T       Calculated noise threshold (can be useful for diagnosing noise
            characteristics of images). Once you know this you can then specify
            fixed thresholds and save some computation time.

    The convolutions are done via the FFT. Many of the parameters relate to
    the specification of the filters in the frequency plane. The values do
    not seem to be very critical and the defaults are usually fine. You may
    want to experiment with the values of 'nscales' and 'k', the noise
    compensation factor.

    Notes on filter settings to obtain even coverage of the spectrum:
    sigmaOnf    .85   mult 1.3
    sigmaOnf    .75   mult 1.6  (filter bandwidth ~1 octave)
    sigmaOnf    .65   mult 2.1
    sigmaOnf    .55   mult 3    (filter bandwidth ~2 octaves)

    For maximum speed the input image should have dimensions that correspond
    to powers of 2, but the code will operate on images of arbitrary size.

    See also:   phasecong, which uses oriented filters and is therefore
                slower, but returns more detailed output

    References:
    ------------
    Peter Kovesi, "Image Features From Phase Congruency". Videre: A Journal of
    Computer Vision Research. MIT Press. Volume 1, Number 3, Summer 1999
    http://mitpress.mit.edu/e-journals/Videre/001/v13.html

    Michael Felsberg and Gerald Sommer, "A New Extension of Linear Signal
    Processing for Estimating Local Properties and Detecting Features". DAGM
    Symposium 2000, Kiel

    Michael Felsberg and Gerald Sommer. "The Monogenic Signal" IEEE
    Transactions on Signal Processing, 49(12):3136-3144, December 2001

    Peter Kovesi, "Phase Congruency Detects Corners and Edges". Proceedings
    DICTA 2003, Sydney Dec 10-12

    """

    if img.dtype not in ['float32', 'float64']:
        img = np.float64(img)
        imgdtype = 'float64'
    else:
        imgdtype = img.dtype

    if img.ndim == 3:
        img = img.mean(2)

    rows, cols = img.shape

    epsilon = 1E-4          # used to prevent /0.
    _, IM = perfft2(img)     # periodic Fourier transform of image

    zeromat = np.zeros((rows, cols), dtype=imgdtype)
    sumAn = zeromat.copy()
    sumf = zeromat.copy()
    sumh1 = zeromat.copy()
    sumh2 = zeromat.copy()

    radius, u1, u2 = filtergrid(rows, cols)

    # Get rid of the 0 radius value at the 0 frequency point (at top-left
    # corner after fftshift) so that taking the log of the radius will not
    # cause trouble.
    radius[0, 0] = 1.

    # Construct the monogenic filters in the frequency domain. The two filters
    # would normally be constructed as follows:
    #    H1 = i*u1./radius
    #    H2 = i*u2./radius
    # However the two filters can be packed together as a complex valued
    # matrix, one in the real part and one in the imaginary part. Do this by
    # multiplying H2 by i and then adding it to H1 (note the subtraction
    # because i*i = -1).  When the convolution is performed via the fft the
    # real part of the result will correspond to the convolution with H1 and
    # the imaginary part with H2. This allows the two convolutions to be done
    # as one in the frequency domain, saving time and memory.
    H = (1j * u1 - u2) / radius

    # The two monogenic filters H1 and H2 are not selective in terms of the
    # magnitudes of the frequencies. The code below generates bandpass log-
    # Gabor filters which are point-wise multiplied by IM to produce different
    # bandpass versions of the image before being convolved with H1 and H2
    #
    # First construct a low-pass filter that is as large as possible, yet falls
    # away to zero at the boundaries. All filters are multiplied by this to
    # ensure no extra frequencies at the 'corners' of the FFT are incorporated
    # as this can upset the normalisation process when calculating phase
    # congruency

    # Updated filter parameters 6/9/2013:   radius .45, 'sharpness' 15
    lp = _lowpassfilter((rows, cols), .45, 15)
    logGaborDenom = 2. * np.log(sigmaOnf) ** 2.

    for ss in range(nscale):
        wavelength = minWaveLength * mult ** ss
        fo = 1. / wavelength  # Centre frequency of filter
        logRadOverFo = (np.log(radius / fo))
        logGabor = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom)
        logGabor *= lp      # Apply the low-pass filter
        logGabor[0, 0] = 0.  # Undo the radius fudge

        IMF = IM * logGabor   # Frequency bandpassed image
        f = np.real(ifft2(IMF))  # Spatially bandpassed image

        # Bandpassed monogenic filtering, real part of h contains
        # convolution result with h1, imaginary part contains
        # convolution result with h2.
        h = ifft2(IMF * H)
        h1, h2 = np.real(h), np.imag(h)

        # Amplitude of this scale component
        An = np.sqrt(f * f + h1 * h1 + h2 * h2)

        # Sum of amplitudes across scales
        sumAn += An
        sumf += f
        sumh1 += h1
        sumh2 += h2

        # At the smallest scale estimate noise characteristics from the
        # distribution of the filter amplitude responses stored in sumAn. tau
        # is the Rayleigh parameter that is used to describe the distribution.
        if ss == 0:
            # Use median to estimate noise statistics
            if noiseMethod == -1:
                tau = np.median(sumAn.flatten()) / np.sqrt(np.log(4))

            # Use the mode to estimate noise statistics
            elif noiseMethod == -2:
                tau = _rayleighmode(sumAn.flatten())

            maxAn = An
        else:
            # Record the maximum amplitude of components across scales to
            # determine the frequency spread weighting
            maxAn = np.maximum(maxAn, An)

        # Form weighting that penalizes frequency distributions that are
        # particularly narrow. Calculate fractional 'width' of the frequencies
        # present by taking the sum of the filter response amplitudes and
        # dividing by the maximum amplitude at each point on the image.   If
        # there is only one non-zero component width takes on a value of 0, if
        # all components are equal width is 1.
        width = (sumAn / (maxAn + epsilon) - 1.) / (nscale - 1)

        # Calculate the sigmoidal weighting function for this
        # orientation
        weight = 1. / (1. + np.exp(g * (cutOff - width)))

        # Automatically determine noise threshold

        # Assuming the noise is Gaussian the response of the filters to noise
        # will form Rayleigh distribution. We use the filter responses at the
        # smallest scale as a guide to the underlying noise level because the
        # smallest scale filters spend most of their time responding to noise,
        # and only occasionally responding to features. Either the median, or
        # the mode, of the distribution of filter responses can be used as a
        # robust statistic to estimate the distribution mean and standard
        # deviation as these are related to the median or mode by fixed
        # constants. The response of the larger scale filters to noise can then
        # be estimated from the smallest scale filter response according to
        # their relative bandwidths.

        # This code assumes that the expected reponse to noise on the phase
        # congruency calculation is simply the sum of the expected noise
        # responses of each of the filters. This is a simplistic overestimate,
        # however these two quantities should be related by some constant that
        # will depend on the filter bank being used. Appropriate tuning of the
        # parameter 'k' will allow you to produce the desired output.

        # fixed noise threshold
        if noiseMethod >= 0:
            T = noiseMethod

        # Estimate the effect of noise on the sum of the filter responses as
        # the sum of estimated individual responses (this is a simplistic
        # overestimate). As the estimated noise response at succesive scales is
        # scaled inversely proportional to bandwidth we have a simple geometric
        # sum.
        else:
            totalTau = tau * (1. - (1. / mult) ** nscale) / (1. - (1. / mult))

            # Calculate mean and std dev from tau using fixed relationship
            # between these parameters and tau. See:
            # <http://mathworld.wolfram.com/RayleighDistribution.html>
            EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2.)
            EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2.)

            # Noise threshold, must be >= epsilon
            T = np.max((EstNoiseEnergyMean + k * EstNoiseEnergySigma, epsilon))

    # Final computation of key quantities
    ori = np.arctan(-sumh2 / sumh1)

    # Wrap angles between -pi and pi and convert radians to degrees
    ori = np.fix((ori % np.pi) / np.pi * 180.)

    # Feature type (a phase angle between -pi/2 and pi/2)
    ft = np.arctan2(sumf, np.sqrt(sumh1 * sumh1 + sumh2 * sumh2))

    # Overall energy
    energy = np.sqrt(sumf * sumf + sumh1 * sumh1 + sumh2 * sumh2)

    # Compute phase congruency. The original measure,
    #
    #   PC = energy/sumAn
    #
    # is proportional to the weighted cos(phasedeviation).  This is not very
    # localised so this was modified to
    #
    #   PC = cos(phasedeviation) - |sin(phasedeviation)|
    # (Note this was actually calculated via dot and cross products.)  This
    # measure approximates
    #
    #   PC = 1 - phasedeviation.
    #
    # However, rather than use dot and cross products it is simpler and more
    # efficient to simply use acos(energy/sumAn) to obtain the weighted phase
    # deviation directly. Note, in the expression below the noise threshold is
    # not subtracted from energy immediately as this would interfere with the
    # phase deviation computation. Instead it is applied as a weighting as a
    # fraction by which energy exceeds the noise threshold. This weighting is
    # applied in addition to the weighting for frequency spread. Note also the
    # phase deviation gain factor which acts to sharpen up the edge response. A
    # value of 1.5 seems to work well. Sensible values are from 1 to about 2.

    phase_dev = np.maximum(
        1. - deviationGain * np.arccos(energy / (sumAn + epsilon)), 0)
    energy_thresh = np.maximum(energy - T, 0)

    M = weight * phase_dev * energy_thresh / (energy + epsilon)

    return M, ori, ft, T
