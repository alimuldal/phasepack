import numpy as np
import scipy as sp
from scipy.fftpack import fftshift, ifftshift
from tools import rayleighmode as _rayleighmode
from tools import lowpassfilter as _lowpassfilter

# Try and use the faster Fourier transform functions from the anfft module if
# available
try:
	import anfft as _anfft
	# measure == True for self-optimisation of repeat Fourier transforms of 
	# similarly-shaped arrays
	def fft2(A):
		return _anfft.fftn(_checkffttype(A),measure=True)
	def ifft2(B):
		return _anfft.ifftn(_checkffttype(B),measure=True)
	def _checkffttype(C):
		# make sure input arrays are typed correctly for FFTW
		if C.dtype == 'complex256':
			# the only incompatible complex type --> complex64
			C = np.complex128(C)
		elif C.dtype not in ['float32','float64','complex64','complex128']:
			# any other incompatible type --> float64
			C = np.float64(C)
		return C
# Otherwise use the normal scipy fftpack ones instead (~2-3x slower!)
except ImportError:
	print \
	"Module 'anfft' (FFTW Python bindings) could not be imported.\n"\
	"To install it, try running 'easy_install anfft' from the terminal.\n"\
	"Falling back on the slower 'fftpack' module for 2D Fourier transforms."
	from scipy.fftpack import fft2, ifft2

def phasecong(	img,
		nscale = 4,
		norient = 6,
		minWaveLength = 3,
		mult = 2.1,
		sigmaOnf = 0.55,
		k = 2.,
		cutOff = 0.5,
		g = 10.,
		noiseMethod = -1):
	"""
	Function for computing phase congruency on an image. This is a contrast-
	invariant edge and corner detector.

	See also: 	phasecongmono, which uses monogenic filters for improved
			speed, but does not return m, PC or EO.

	Arguments:
	------------------------
	<Name>	<Default>	<Description>
	img 		N/A 	The input image
	nscales		5 	Number of wavelet scales, try values 3-6
	norient		6 	Number of filter orientations.
	minWaveLength	3 	Wavelength of smallest scale filter.
	mult 		2.1 	Scaling factor between successive filters.
	sigmaOnf 	0.55 	Ratio of the standard deviation of the Gaussian 
				describing the log Gabor filter's transfer function 
				in the frequency domain to the filter center frequency.
	k 		2.0 	No of standard deviations of the noise energy beyond
				the mean at which we set the noise threshold point.
				You may want to vary this up to a value of 10 or
				20 for noisy images
	cutOff 		0.5 	The fractional measure of frequency spread below
				which phase congruency values get penalized
	g		10 	Controls the 'sharpness' of the transition in the 
				sigmoid function used to weight phase congruency
				for frequency spread
	noiseMethod 	-1 	Parameter specifies method used to determine
				noise statistics. 
				-1 use median of smallest scale filter responses
				-2 use mode of smallest scale filter responses
				0+ use noiseMethod value as the fixed noise threshold.

	Return values:
	------------------------
	M 			Maximum moment of phase congruency covariance, 
				which can be used as a measure of edge strength
	m 			Minimum moment of phase congruency covariance, 
				which can be used as a measure of corner strength
	ori 			Orientation image, in integer degrees (0-180), 
				positive angles anti-clockwise.
	ft 			Local weighted mean phase angle at every point in
				the image. A value of pi/2 corresponds to a bright 
				line, 0 to a step and -pi/2 to a dark line.
	PC 			A list of phase congruency images (values between
				0 and 1), one per orientation.
	EO 			A list containing the complex-valued convolution 
				results (see below)
	T 			Calculated noise threshold (can be useful for
				diagnosing noise characteristics of images).  Once you know
				this you can then specify fixed thresholds and save some
				computation time.

	EO is a list of sublists, where an entry in the outer list corresponds to 
	a spatial scale, and an entry in the sublist corresponds to an orientation,
	i.e. EO[o][s] is the result for orientation o and spatial scale s. The real 
	and imaginary parts are the results of convolving with the even and odd 
	symmetric filters respectively.

	Hence:
		abs(E[o][s]) returns the magnitude of the convolution, and
		angle(E[o][s]) returns the phase angles for orientation o and 
		scale s.

	The convolutions are done via the FFT. Many of the parameters relate to
	the specification of the filters in the frequency plane. The values do
	not seem to be very critical and the defaults are usually fine. You may
	want to experiment with the values of 'nscales' and 'k', the noise
	compensation factor.

	Notes on filter settings to obtain even coverage of the spectrum
	sigmaOnf 	.85   mult 1.3
	sigmaOnf 	.75   mult 1.6	(filter bandwidth ~1 octave)
	sigmaOnf 	.65   mult 2.1  
	sigmaOnf 	.55   mult 3 	(filter bandwidth ~2 octaves)

	For maximum speed the input image should have dimensions that correspond
	to powers of 2, but the code will operate on images of arbitrary size.

	Original MATLAB version by Peter Kovesi
	<http://www.csse.uwa.edu.au/~pk/research/matlabfns/PhaseCongruency/phasecong3.m>

	Translated to Python by Alistair Muldal
	<alistair muldal@pharm ox ac uk>

	References:
		Peter Kovesi, "Image Features From Phase Congruency". Videre: A
		Journal of Computer Vision Research. MIT Press. Volume 1, Number 3,
		Summer 1999 http://mitpress.mit.edu/e-journals/Videre/001/v13.html

		Peter Kovesi, "Phase Congruency Detects Corners and
		Edges". Proceedings DICTA 2003, Sydney Dec 10-12
	"""

	"""
	Permission is hereby  granted, free of charge, to any  person obtaining a copy
	of this software and associated  documentation files (the "Software"), to deal
	in the Software without restriction, subject to the following conditions:
	 
	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.
	 
	The software is provided "as is", without warranty of any kind.

	"""

	if img.dtype not in ['float32', 'float64']:
		img = np.float64(img)
		imgdtype = 'float64'
	else:
		imgdtype = img.dtype

	if img.ndim == 3:
		img = img.mean(2)

	rows,cols = img.shape

	epsilon = 1E-4		# used to prevent /0.
	IM = fft2(img)	# Fourier transformed image

	# lists to contain convolution results and phase congruency images
	EO = []
	PC = []

	zeromat = np.zeros((rows,cols),dtype=imgdtype)

	# matrices for covariance data
	covx2 = zeromat.copy()
	covy2 = zeromat.copy()
	covxy = zeromat.copy()

	# Total energy vectors for finding feature orientation and type
	EnergyV = np.zeros((rows,cols,3),dtype=imgdtype)

	# Sum of phase congruences across orientations
	pcSum = zeromat.copy()

	# Pre-compute some stuff to speed up filter construction

	# Set up X and Y matrices with ranges normalised to +/- 0.5
	y,x = np.ogrid[-0.5:0.5:(1./rows),-0.5:0.5:(1./cols)]

	# normalised distance from centre
	radius = np.sqrt(x**2. + y**2.)

	# polar angle (-ve y gives +ve anti-clockwise angles)
	theta = np.arctan2(-y,x)

	# Quadrant shift radius and theta so that filters are constructed with 0
	# frequency at the corners
	radius = fftshift(radius)
	theta = fftshift(theta)

	# Get rid of the 0 radius value at the 0 frequency point (now at top-left
	# corner) so that taking the log of the radius will not cause trouble.
	radius[0,0] = 1.

	sintheta = np.sin(theta)
	costheta = np.cos(theta)

	del x, y, theta

	# Construct a bank of log-Gabor filters at different spatial scales

	# Filters are constructed in terms of two components.
	# 1) The radial component, which controls the frequency band that the filter
	#    responds to
	# 2) The angular component, which controls the orientation that the filter
	#    responds to.
	# The two components are multiplied together to construct the overall filter.

	# Construct the radial filter components... First construct a low-pass
	# filter that is as large as possible, yet falls away to zero at the
	# boundaries. All log Gabor filters are multiplied by this to ensure no
	# extra frequencies at the 'corners' of the FFT are incorporated as this
	# seems to upset the normalisation process when calculating phase
	# congrunecy.
	lp = _lowpassfilter([rows,cols],.4,10);	# Radius .4, 'sharpness' 10

	logGaborDenom = 2.*np.log(sigmaOnf)**2.
	logGabor = []

	for ss in xrange(nscale):
		wavelength = minWaveLength*mult**(ss)

		# centre of frequency filter
		fo = 1./wavelength

		# log Gabor
		tmp = np.exp( (-(np.log(radius/fo))**2.)/logGaborDenom )

		# apply low-pass filter
		tmp = tmp*lp

		# set the value at the 0 frequency point of the filter back to
		# zero (undo the radius fudge).
		tmp[0,0] = 0.

		logGabor.append(tmp)

	# MAIN LOOP
	# for each orientation...
	for oo in xrange(norient):

		# Construct the angular filter spread function
		angl = oo*(np.pi/norient)

		# For each point in the filter matrix calculate the angular
		# distance from the specified filter orientation.  To overcome
		# the angular wrap-around problem sine difference and cosine
		# difference values are first computed and then the arctan2
		# function is used to determine angular distance.

		# difference in sine and cosine
		ds = sintheta*np.cos(angl) - costheta*np.sin(angl)
		dc = costheta*np.cos(angl) + sintheta*np.sin(angl)

		# absolute angular difference
		dtheta = np.abs(np.arctan2(ds,dc))

		# Scale theta so that cosine spread function has the right
		# wavelength and clamp to pi.
		dtheta *= norient/2.
		np.where(dtheta <= np.pi,dtheta,np.pi)

		# The spread function is cos(dtheta) between -pi and pi.  We add
		# 1, and then divide by 2 so that the value ranges 0-1
		spread = (np.cos(dtheta)+1.)/2.

		# Initialize accumulators
		sumE_ThisOrient = zeromat.copy()
		sumO_ThisOrient = zeromat.copy()
		sumAn_ThisOrient  = zeromat.copy()
		Energy = zeromat.copy()

		EOscale = []

		# for each scale...
		for ss in xrange(nscale):

			# Multiply radial and angular components to get filter
			filt = logGabor[ss]*spread

			# Convolve image with even and odd filters
			thisEO = ifft2(IM*filt)	# <-- here's where the MATLAB version differs!

			# Even + odd filter response amplitude
			An = np.abs(thisEO)

			# Sum of amplitudes for even & odd filters across scales
			sumAn_ThisOrient += An

			# Sum of even filter convolution results
			sumE_ThisOrient += np.real(thisEO)

			# Sum of odd filter convolution results
			sumO_ThisOrient += np.imag(thisEO)

			# At the smallest scale estimate noise characteristics
			# from the distribution of the filter amplitude
			# responses stored in sumAn.  tau is the Rayleigh
			# parameter that is used to describe the distribution.
			if ss == 0:
				# Use median to estimate noise statistics
				if noiseMethod == -1:
					tau = np.median(sumAn_ThisOrient.flatten())/np.sqrt(np.log(4))

				# Use the mode to estimate noise statistics
				elif noiseMethod == -2:
					tau = _rayleighmode(sumAn_ThisOrient.flatten())

				maxAn = An
			else:
				# Record the maximum amplitude of components
				# across scales to determine the frequency
				# spread weighting
				maxAn = np.where(maxAn>An,maxAn,An)

			# append per-scale list
			EOscale.append(thisEO)

		# Accumulate total 3D energy vector data, this will be used to
		# determine overall feature orientation and feature phase/type
		EnergyV[:,:,0] += sumE_ThisOrient
		EnergyV[:,:,1] += np.cos(angl)*sumO_ThisOrient
		EnergyV[:,:,2] += np.sin(angl)*sumO_ThisOrient

		# Get weighted mean filter response vector, this gives the
		# weighted mean phase angle.
		XEnergy = np.sqrt(sumE_ThisOrient**2. + sumO_ThisOrient**2) + epsilon
		MeanE = sumE_ThisOrient / XEnergy
		MeanO = sumO_ThisOrient / XEnergy

		# Now calculate An(cos(phase_deviation)-| sin(phase_deviation))|
		# by using dot and cross products between the weighted mean
		# filter response vector and the individual filter response
		# vectors at each scale.  This quantity is phase congruency
		# multiplied by An, which we call energy.
		for ss in xrange(nscale):
			E = np.real(EOscale[ss])
			O = np.imag(EOscale[ss])
			Energy += E*MeanE + O*MeanO - np.abs(E*MeanO - O*MeanE)

		# Automatically determine noise threshold

		# Assuming the noise is Gaussian the response of the filters to noise will
		# form Rayleigh distribution.  We use the filter responses at the smallest
		# scale as a guide to the underlying noise level because the smallest
		# scale filters spend most of their time responding to noise, and only
		# occasionally responding to features. Either the median, or the mode, of
		# the distribution of filter responses can be used as a robust statistic
		# to estimate the distribution mean and standard deviation as these are
		# related to the median or mode by fixed constants.  The response of the
		# larger scale filters to noise can then be estimated from the smallest
		# scale filter response according to their relative bandwidths.

		# This code assumes that the expected reponse to noise on the phase
		# congruency calculation is simply the sum of the expected noise responses
		# of each of the filters.  This is a simplistic overestimate, however
		# these two quantities should be related by some constant that will depend
		# on the filter bank being used.  Appropriate tuning of the parameter 'k'
		# will allow you to produce the desired output.

		# fixed noise threshold
		if noiseMethod >= 0:
			T = noiseMethod

		# Estimate the effect of noise on the sum of the filter responses as
		# the sum of estimated individual responses (this is a simplistic
		# overestimate). As the estimated noise response at succesive scales
		# is scaled inversely proportional to bandwidth we have a simple
		# geometric sum.
		else:
			totalTau = tau*(1. - (1./mult)**nscale)/(1. - (1./mult))

			# Calculate mean and std dev from tau using fixed relationship
			# between these parameters and tau. See
			# <http://mathworld.wolfram.com/RayleighDistribution.html>
			EstNoiseEnergyMean = totalTau*np.sqrt(np.pi/2.)
			EstNoiseEnergySigma = totalTau*np.sqrt((4-np.pi)/2.)

			# Noise threshold, must be >= epsilon
			T = np.max((EstNoiseEnergyMean + k*EstNoiseEnergySigma,epsilon))

		# Apply noise threshold, this is effectively wavelet denoising
		# via soft thresholding.
		Energy = np.where(Energy-T >= 0, Energy-T, 0)

		# Form weighting that penalizes frequency distributions that are
		# particularly narrow.  Calculate fractional 'width' of the
		# frequencies present by taking the sum of the filter response
		# amplitudes and dividing by the maximum amplitude at each point
		# on the image.   If there is only one non-zero component width
		# takes on a value of 0, if all components are equal width is 1.
		width = (sumAn_ThisOrient/(maxAn+epsilon)-1.)/(nscale-1)

		# Calculate the sigmoidal weighting function for this
		# orientation
		weight = 1./(1.+np.exp(g*(cutOff-width)))

		# Apply weighting to energy, then calculate phase congruency
		thisPC = weight*Energy/sumAn_ThisOrient
		pcSum += thisPC

		# accumulate covariance data
		covx = thisPC*np.cos(angl)
		covy = thisPC*np.sin(angl)
		covx2 += covx**2.
		covy2 += covy**2.
		covxy += covx*covy

		# append per-orientation lists
		PC.append(thisPC)
		EO.append(EOscale)

	# Edge and Corner calculations
	# The following is optimised code to calculate principal vectors of the
	# phase congruency covariance data and to calculate the minimumum and
	# maximum moments - these correspond to the singular values.

	# First normalise covariance values by the number of orientations/2
	covx2 /= norient/2.
	covy2 /= norient/2.
	covxy *= 4./norient	# This gives us 2*covxy/(norient/2)
	denom = np.sqrt(covxy**2.+(covx2-covy2)**2.) + epsilon

	# Maximum and minimum moments
	M = (covx2+covy2 + denom)/2.
	m = (covx2+covy2 - denom)/2.

	# Orientation and feature phase/type
	ori = np.arctan2(EnergyV[:,:,2],EnergyV[:,:,1])

	# Wrap angles between -pi and pi and convert radians to degrees
	ori = np.where(ori < 0,ori+np.pi,ori)
	ori = np.round(ori*180./np.pi)

	OddV = np.sqrt( EnergyV[:,:,1]**2. + EnergyV[:,:,2]**2. )

	# Feature phase pi/2 --> white line, 0 --> step, -pi/2 --> black line
	ft = np.arctan2(EnergyV[:,:,0],OddV)

	return M,m,ori,ft,PC,EO,T