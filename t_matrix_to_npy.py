#!/usr/bin/env python

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

import argh


def main(input_file=None, output_file=None, frequency_file=None):
    f, _, _, _, _, t11, t12, t21, t22 = np.loadtxt(input_file, dtype=complex, unpack=True)
    f = f.view(complex)
    t_matrix = []
    for t11n, t12n, t21n, t22n in zip(t11, t12, t21, t22):
        t_matrix.append([[t11n, t12n],[t21n, t22n]])

    T = np.asarray(t_matrix)

    print "T.shape", T.shape

    if not frequency_file:
        frequency_file = "frequency_" + output_file

    np.save(frequency_file, f)
    np.save(output_file, T)

if __name__ == '__main__':
     argh.dispatch_command(main)
