import numpy as np
import scipy as sp
from scipy.fftpack import fftshift, ifftshift

def lowpassfilter(size,cutoff,n):
	"""
	Constructs a low-pass Butterworth filter:

	     f =    1/(1.0 + (w/cutoff)^2n)

	usage: f = _lowpassfilter(sze, cutoff, n)

	where: size   is a tuple specifying the size of filter to construct
	              [rows cols].
	       cutoff is the cutoff frequency of the filter 0 - 0.5
	       n      is the order of the filter, the higher n is the sharper
	              the transition is. (n must be an integer >= 1).
	              Note that n is doubled so that it is always an even integer.

	The frequency origin of the returned filter is at the corners.
	"""

	if cutoff < 0. or cutoff > 0.5:
		raise Exception('cutoff must be between 0 and 0.5')
	elif n % 1:
		raise Exception('n must be an integer >= 1')
	if len(size) == 1:
		rows = size; cols = size
	else:
		rows,cols = size

	x,y = np.ogrid[-0.5:0.5:(1./rows),-0.5:0.5:(1./cols)]
	radius = np.sqrt(x**2.+y**2.)
	f = ifftshift(1. / (1. + (radius/cutoff)**(2.*n)))

	return f

def rayleighmode(data,nbins=50):
	"""
	Computes mode of a vector/matrix of data that is assumed to come from a
	Rayleigh distribution.

	Usage:  rmode = _rayleighmode(data, nbins)

	Arguments:  data  - data assumed to come from a Rayleigh distribution
	            nbins - Optional number of bins to use when forming histogram
	                    of the data to determine the mode.

	Mode is computed by forming a histogram of the data over 50 bins and then
	finding the maximum value in the histogram.  Mean and standard deviation
	can then be calculated from the mode as they are related by fixed
	constants.

	mean = mode * sqrt(pi/2)
	std dev = mode * sqrt((4-pi)/2)

	See
	<http://mathworld.wolfram.com/RayleighDistribution.html>
	<http://en.wikipedia.org/wiki/Rayleigh_distribution>
	"""
	
	n,edges = np.histogram(data,nbins)
	ind = np.argmax(n)
	rmode = (edges[ind]+edges[ind+1])/2.

	return rmode