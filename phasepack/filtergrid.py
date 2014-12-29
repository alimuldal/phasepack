import numpy as np
from scipy.fftpack import ifftshift


def filtergrid(rows, cols):
	# Set up u1 and u2 matrices with ranges normalised to +/- 0.5
	if (cols % 2):
		xvals = np.arange(-(cols-1)/2., ((cols-1)/2.)+1) / float(cols-1)
	else:
		xvals = np.arange(-cols/2., cols/2.) / float(cols)

	if (rows % 2):
		yvals = np.arange(-(rows-1)/2., ((rows-1)/2.)+1) / float(rows-1)
	else:
		yvals = np.arange(-rows/2., rows/2.) / float(rows)
	u1,u2 = np.meshgrid(xvals,yvals,sparse=True)

	# Quadrant shift to put 0 frequency at the corners
	u1 = ifftshift(u1)
	u2 = ifftshift(u2)

	# Compute frequency values as a radius from centre (but quadrant shifted)
	radius = np.sqrt(u1*u1 + u2*u2)

	return radius, u1, u2
