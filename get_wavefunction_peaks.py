#!/usr/bin/env python2.7

from glob import glob
import numpy as np
import matplotlib.pyplot as plt

import argh

from ep.helpers import get_local_peaks, get_local_minima
from ep.potential import gauss
from helper_functions import convert_to_complex


def get_array(input, L=100, W=1, r_nx=None, r_ny=None, abs=True):
    n, psi = np.genfromtxt(input, unpack=True, usecols=(0,1), dtype=complex,
                           converters={1: convert_to_complex})
    x = np.linspace(0, L, r_nx)
    y = np.linspace(0, W, r_ny)
    X, Y = np.meshgrid(x,y)
    Z = [ x.reshape(r_ny, r_nx, order='F') for x in psi, ]

    if abs:
        Z = abs(Z)**2

    return X, Y, Z


def main(pphw=50, N=2.5, L=100, W=1, eps=0.1, sigma=0.01, plot=True):
    nyout = pphw*N + 1.
    r_nx_pot = int(nyout*L)
    r_ny_pot = int(nyout*W)
    print "r_nx_pot, r_ny_pot", r_nx_pot, r_ny_pot

    array_kwargs = {'L': L,
                    'W': W,
                    'r_nx': r_nx_pot,
                    'r_ny': r_ny_pot,
                    'eps': eps}
    X, Y, Za = get_array(glob("pic.*.0000.streu.*.ascii")[0], **array_kwargs)
    _, _, Zb = get_array(glob("pic.*.0001.streu.*.ascii")[0], **array_kwargs)

    peaks_a, peaks_b = [ get_local_peaks(z, peak_type='minimum') for z in Za, Zb ]
    idx_a, idx_b = [ np.where(p) for p in peaks_a, peaks_b ]

    if plot:
        f, (ax1, ax2) = plt.subplots(nrows=2, figsize=(200, 100))
        cmap = plt.cm.jet
        cmap.set_bad('dimgrey', 1)

        scale = 5.
        ax1.pcolormesh(X, Y, scale*Zma, cmap=cmap)
        ax1.scatter(X[idx_b], Y[idx_b], s=1.5e4, c="w", edgecolors=None)

        ax2.pcolormesh(X, Y, scale*Zmb, cmap=cmap)
        ax2.scatter(X[idx_b], Y[idx_b], s=1.5e4, c="w", edgecolors=None)

        for ax in (ax1, ax2):
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())

        plt.savefig('wavefunction.jpg', bbox_inches='tight')

    Z_pot = np.zeros_like(X)

    for (xn, yn) in zip(X[idx_b].flatten(), Y[idx_b].flatten()):
            Z_pot -= gauss(X, xn, sigma) * gauss(Y, yn, sigma)

    np.savetxt("output_potential.dat",
               zip(range(len(Z_pot.flatten('F'))), Z_pot.flatten('F')))
    print "output_potential.dat written."

    np.save("output_potential.npy", Z_pot)


if __name__ == '__main__':
    argh.dispatch_command(main)
