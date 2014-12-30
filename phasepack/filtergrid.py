import numpy as np
from scipy.fftpack import ifftshift


def filtergrid(rows, cols):

    # Set up u1 and u2 matrices with ranges normalised to +/- 0.5
    u1, u2 = np.meshgrid(np.linspace(-0.5, 0.5, cols, endpoint=(cols % 2)),
                         np.linspace(-0.5, 0.5, rows, endpoint=(rows % 2)),
                         sparse=True)

    # Quadrant shift to put 0 frequency at the top left corner
    u1 = ifftshift(u1)
    u2 = ifftshift(u2)

    # Compute frequency values as a radius from centre (but quadrant shifted)
    radius = np.sqrt(u1 * u1 + u2 * u2)

    return radius, u1, u2
