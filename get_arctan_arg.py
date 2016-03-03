#!/usr/bin/env python

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

import argh

from ep.plot import get_colors

colors, _, _ = get_colors()

def plot_coefficients(evecs_file=None):
    (ev1_abs, ev1_phi, v1r, v1i, v2r, v2i,
     ev2_abs, ev2_phi, c1r, c1i, c2r, c2i) = np.loadtxt(evecs_file).T
    v1 = v1r + 1j*v1i
    v2 = v2r + 1j*v2i
    c1 = c1r + 1j*c1i
    c2 = c2r + 1j*c2i

    print "np.arctan(abs(v1/v2))", np.arctan(abs(v1/v2))
    print "np.arctan(abs(c1/c2))", np.arctan(abs(c1/c2))
    print "np.angle(v1/v2)", np.angle(v1/v2)
    print "np.angle(c1/c2)", np.angle(c1/c2)

if __name__ == '__main__':
    argh.dispatch_command(plot_coefficients)
