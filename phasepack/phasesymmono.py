# MIT License:

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# The software is provided "as is", without warranty of any kind.

# Original MATLAB version by Peter Kovesi
# <http://www.csse.uwa.edu.au/~pk/research/matlabfns/PhaseCongruency/phasesym.m>

# Python translation by Alistair Muldal
# <alistair muldal@pharm ox ac uk>


import numpy as np
from scipy.fftpack import fftshift, ifftshift

from .tools import rayleighmode as _rayleighmode
from .tools import lowpassfilter as _lowpassfilter
from .filtergrid import filtergrid

# Try and use the faster Fourier transform functions from the pyfftw module if
# available
from .tools import fft2, ifft2


def phasesymmono(img, nscale=5, minWaveLength=3, mult=2.1, sigmaOnf=0.55, k=2.,
                 polarity=0, noiseMethod=-1):
    """
    This function calculates the phase symmetry of points in an image. This is
    a contrast invariant measure of symmetry. This function can be used as a
    line and blob detector. The greyscale 'polarity' of the lines that you want
    to find can be specified.

    Arguments:
    -----------
    <Name>      <Default>   <Description>
    img             N/A     The input image
    nscale          5       Number of wavelet scales, try values 3-6
    minWaveLength   3       Wavelength of smallest scale filter.
    mult            2.1     Scaling factor between successive filters.
    sigmaOnf        0.55    Ratio of the standard deviation of the Gaussian
                            describing the log Gabor filter's transfer function
                            in the frequency domain to the filter center
                            frequency.
    k               2.0     No. of standard deviations of the noise energy
                            beyond the mean at which we set the noise threshold
                            point. You may want to vary this up to a value of
                            10 or 20 for noisy images.
    polarity        0       Controls 'polarity' of symmetry features to find.
                            1 only return 'bright' features
                            -1 only return 'dark' features
                            0 return both 'bright' and 'dark' features
    noiseMethod     -1      Parameter specifies method used to determine
                            noise statistics.
                            -1 use median of smallest scale filter responses
                            -2 use mode of smallest scale filter responses
                            >=0 use this value as the fixed noise threshold

    Returns:
    ---------
    phaseSym        Phase symmetry image (values between 0 and 1).
    totalEnergy     Un-normalised raw symmetry energy which may be more to your
                    liking.
    T               Calculated noise threshold (can be useful for diagnosing
                    noise characteristics of images). Once you know this you
                    can then specify fixed thresholds and save some computation
                    time.

    The convolutions are done via the FFT. Many of the parameters relate to the
    specification of the filters in the frequency plane. The values do not seem
    to be very critical and the defaults are usually fine. You may want to
    experiment with the values of 'nscales' and 'k', the noise compensation
    factor.

    Notes on filter settings to obtain even coverage of the spectrum
    sigmaOnf    .85   mult 1.3
    sigmaOnf    .75   mult 1.6  (filter bandwidth ~1 octave)
    sigmaOnf    .65   mult 2.1
    sigmaOnf    .55   mult 3    (filter bandwidth ~2 octaves)

    For maximum speed the input image should have dimensions that correspond to
    powers of 2, but the code will operate on images of arbitrary size.

    See also:   phasesym, which uses oriented filters and is therefore
                slower, but also returns an orientation map of the image

    References:
    ------------
    Peter Kovesi, "Symmetry and Asymmetry From Local Phase" AI'97, Tenth
    Australian Joint Conference on Artificial Intelligence. 2 - 4 December
    1997. http://www.cs.uwa.edu.au/pub/robvis/papers/pk/ai97.ps.gz.

    Peter Kovesi, "Image Features From Phase Congruency". Videre: A Journal of
    Computer Vision Research. MIT Press. Volume 1, Number 3, Summer 1999
    http://mitpress.mit.edu/e-journals/Videre/001/v13.html

    Michael Felsberg and Gerald Sommer, "A New Extension of Linear Signal
    Processing for Estimating Local Properties and Detecting Features". DAGM
    Symposium 2000, Kiel

    Michael Felsberg and Gerald Sommer. "The Monogenic Signal" IEEE
    Transactions on Signal Processing, 49(12):3136-3144, December 2001

    """

    if img.dtype not in ['float32', 'float64']:
        img = np.float64(img)
        imgdtype = 'float64'
    else:
        imgdtype = img.dtype

    if img.ndim == 3:
        img = img.mean(2)
    rows, cols = img.shape

    epsilon = 1E-4  # used to prevent /0.
    IM = fft2(img)  # Fourier transformed image

    zeromat = np.zeros((rows, cols), dtype=imgdtype)

    # Matrix for accumulating weighted phase congruency values (energy).
    totalEnergy = zeromat.copy()

    # Matrix for accumulating filter response amplitude values.
    sumAn = zeromat.copy()

    radius, u1, u2 = filtergrid(rows, cols)

    # Get rid of the 0 radius value at the 0 frequency point (at top-left
    # corner after fftshift) so that taking the log of the radius will not
    # cause trouble.
    radius[0, 0] = 1.

    # Construct the monogenic filters in the frequency domain. The two
    # filters would normally be constructed as follows
    #    H1 = i*u1./radius
    #    H2 = i*u2./radius
    # However the two filters can be packed together as a complex valued
    # matrix, one in the real part and one in the imaginary part. Do this by
    # multiplying H2 by i and then adding it to H1 (note the subtraction
    # because i*i = -1). When the convolution is performed via the fft the real
    # part of the result will correspond to the convolution with H1 and the
    # imaginary part with H2. This allows the two convolutions to be done as
    # one in the frequency domain, saving time and memory.
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
    lp = _lowpassfilter([rows, cols], .4, 10)
    # Radius .4, 'sharpness' 10
    logGaborDenom = 2. * np.log(sigmaOnf) ** 2.

    for ss in range(nscale):
        wavelength = minWaveLength * mult ** ss
        fo = 1. / wavelength  # Centre frequency of filter

        logRadOverFo = np.log(radius / fo)
        logGabor = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom)
        logGabor *= lp      # Apply the low-pass filter
        logGabor[0, 0] = 0.  # Undo the radius fudge

        IMF = IM * logGabor   # Frequency bandpassed image
        f = np.real(ifft2(IMF))  # Spatially bandpassed image

        # Bandpassed monogenic filtering, real part of h contains convolution
        # result with h1, imaginary part contains convolution result with h2.
        h = ifft2(IMF * H)

        # Squared amplitude of the h1 and h2 filters
        hAmp2 = h.real * h.real + h.imag * h.imag

        # Magnitude of energy
        sumAn += np.sqrt(f * f + hAmp2)

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

        # Calculate the phase symmetry measure

        # look for 'white' and 'black' spots
        if polarity == 0:
            totalEnergy += np.abs(f) - np.sqrt(hAmp2)

        # just look for 'white' spots
        elif polarity == 1:
            totalEnergy += f - np.sqrt(hAmp2)

        # just look for 'black' spots
        elif polarity == -1:
            totalEnergy += -f - np.sqrt(hAmp2)

    # Automatically determine noise threshold

    # Assuming the noise is Gaussian the response of the filters to noise will
    # form Rayleigh distribution. We use the filter responses at the smallest
    # scale as a guide to the underlying noise level because the smallest scale
    # filters spend most of their time responding to noise, and only
    # occasionally responding to features. Either the median, or the mode, of
    # the distribution of filter responses can be used as a robust statistic to
    # estimate the distribution mean and standard deviation as these are
    # related to the median or mode by fixed constants. The response of the
    # larger scale filters to noise can then be estimated from the smallest
    # scale filter response according to their relative bandwidths.

    # This code assumes that the expected reponse to noise on the phase
    # congruency calculation is simply the sum of the expected noise responses
    # of each of the filters. This is a simplistic overestimate, however these
    # two quantities should be related by some constant that will depend on the
    # filter bank being used. Appropriate tuning of the parameter 'k' will
    # allow you to produce the desired output.

    # fixed noise threshold
    if noiseMethod >= 0:
        T = noiseMethod

    # Estimate the effect of noise on the sum of the filter responses as the
    # sum of estimated individual responses (this is a simplistic
    # overestimate). As the estimated noise response at succesive scales is
    # scaled inversely proportional to bandwidth we have a simple geometric
    # sum.
    else:
        totalTau = tau * (1. - (1. / mult) ** nscale) / (1. - (1. / mult))

        # Calculate mean and std dev from tau using fixed relationship
        # between these parameters and tau. See
        # <http://mathworld.wolfram.com/RayleighDistribution.html>
        EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2.)
        EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2.)

        # Noise threshold, must be >= epsilon
        T = np.maximum(EstNoiseEnergyMean + k * EstNoiseEnergySigma,
                       epsilon)

    # Apply noise threshold - effectively wavelet denoising soft thresholding
    # and normalize symmetryEnergy by the sumAn to obtain phase symmetry. Note
    # the flooring operation is not necessary if you are after speed, it is
    # just 'tidy' not having -ve symmetry values
    phaseSym = np.maximum(totalEnergy - T, 0)
    phaseSym /= sumAn + epsilon

    return phaseSym, totalEnergy, T
