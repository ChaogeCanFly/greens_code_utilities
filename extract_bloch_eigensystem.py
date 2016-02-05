#!/usr/bin/env python

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

import argh

import bloch


def get_bloch_eigensystem():
    """docstring for get_bloch_eigensystem"""
    # get Bloch eigensystem
    K, _, ev, _, v, _ = bloch.get_eigensystem(return_eigenvectors=True,
                                              return_velocities=True,
                                              verbose=True, neumann=False)
    K0, K1 = K[0], K[1]
    ev0, ev1 = ev[0,:], ev[1,:]
    print "chi.shape", ev0.shape

    plt.plot(np.abs(ev0)**2, "r-")
    plt.plot(np.abs(ev1)**2, "g-")
    plt.show()
    return

    z = ev0.view(dtype=float)
    np.savetxt("eigensystem.dat", zip(ev0.real, ev0.imag,
                                        np.ones_like(z)*K0.real,
                                        np.ones_like(z)*K0.imag,
                                        ev1.real, ev1.imag,
                                        np.ones_like(z)*K1.real,
                                        np.ones_like(z)*K1.imag),
                header=('y Re(ev0) Im(ev0) Re(K0) Im(K0) Re(ev1)'
                        'Im(ev1) Re(K1) Im(K1)'))


if __name__ == '__main__':
     argh.dispatch_command(get_bloch_eigensystem)
