from .phasecong import phasecong
from .phasecongmono import phasecongmono
from .phasesym import phasesym
from .phasesymmono import phasesymmono

__all__ = [	'phasesym',
		'phasesymmono',
		'phasecong',
		'phasecongmono'
				]

__doc__ = \
"""
Phasepack - a toolkit for phase-based feature detection
-------------------------------------------------------

This toolkit consists of a set of functions which use information contained
within the phase of a Fourier-transformed image to detect localised features
such as edges, blobs and corners. These methods have the key advantage that the
properties they measure are invariant with respect to image brightness and
contrast.

	phasecong 	Phase congruency using oriented filters
	phasecongmono	Fast phase congruency using monogenic filters
	phasesym 	Phase symmetry using oriented filters
	phasesymmono	Fast phase symmetry using monogenic filters

For more information on a particular function, see the associated docstring and
the references therein.

Fast(er) Fourier Transforms:
All of the functions in this module make use of the Fast Fourier Transform
(FFT), and their speed significantly depends on the module used to provide FFT
functions. If it is available, the 'anfft' module will be used. This provides
Python bindings to the FFTW C library, and is substantially faster than
'fftpack', the default for scipy. To install anfft, run 'easy_install anfft'
from a terminal session.

Authorship:
All of these functions were originally written for MATLAB by Peter Kovesi, and
were ported to Python by Alistair Muldal. The original MATLAB code, as well as
further explanatory information and references are available from Peter Kovesi's
website:

<http://www.csse.uwa.edu.au/~pk/Research/MatlabFns/index.html#phasecong>

MIT License:
Permission is hereby  granted, free of charge, to any  person obtaining a copy
of this software and associated  documentation files (the "Software"), to deal
in the Software without restriction, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

The software is provided "as is", without warranty of any kind.

"""
