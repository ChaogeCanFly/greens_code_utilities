#!/usr/bin/env python

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

import argh


def main(infile=None, outfile=None, freqfile=None):
    f, _, _, _, _, t11, t12, t21, t22 = np.loadtxt(infile, dtype=complex, unpack=True)
    f = f.view(complex)
    t_matrix = []
    for t11n, t12n, t21n, t22n in zip(t11, t12, t21, t22):
        t_matrix.append([[t11n, t12n],[t21n, t22n]])

    T = np.asarray(t_matrix)

    print "T.shape", T.shape

    np.save(freqfile, f)
    np.save(outfile, T)

if __name__ == '__main__':
     argh.dispatch_command(main)
