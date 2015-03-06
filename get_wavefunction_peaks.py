#!/usr/bin/env python2.7

from glob import glob
import numpy as np
import matplotlib.pyplot as plt

import argh

from ascii_to_numpy import read_ascii_array
from ep.helpers import get_local_peaks, get_local_minima
from ep.potential import gauss
from helper_functions import convert_to_complex


# def get_array(input, L=100, W=1, r_nx=None, r_ny=None):
#     n, psi = np.genfromtxt(input, unpack=True, usecols=(0,1), dtype=complex,
#                            converters={1: convert_to_complex})
#     x = np.linspace(0, L, r_nx)
#     y = np.linspace(0, W, r_ny)
#     X, Y = np.meshgrid(x,y)
#     Z = psi.reshape(r_ny, r_nx, order='F')
#     Z = np.abs(Z)**2
#
#     return X, Y, Z


@argh.arg('--mode1', type=str)
@argh.arg('--mode2', type=str)
def main(pphw=50, N=2.5, L=100, W=1, eps=0.1, sigma=0.01, plot=True,
         pic_ascii=False, mode1=None, mode2=None):

    ascii_array_kwargs = {'L': L,
                          'W': W,
                          'pphw': pphw,
                          'N': N,
                          'pic_ascii': pic_ascii,
                          'return_abs': True}
    if mode1 is None:
        mode1 = glob("*.0000.streu.*.purewavefunction.ascii")[0]
    if mode2 is None:
        mode2 = glob("*.0001.streu.*.purewavefunction.ascii")[0]

    # nyout = pphw*N + 1.
    # r_nx_pot = int(nyout*L)
    # r_ny_pot = int(nyout*W)
    # print "r_nx_pot, r_ny_pot", r_nx_pot, r_ny_pot
    #
    # array_kwargs = {'L': L,
    #                 'W': W,
    #                 'r_nx': r_nx_pot,
    #                 'r_ny': r_ny_pot}
    #
    # if mode1 is None:
    #     mode1 = glob("*.0000.streu.*.purewavefunction.ascii")[0]
    # if mode2 is None:
    #     mode2 = glob("*.0001.streu.*.purewavefunction.ascii")[0]
    #
    # X, Y, Za = get_array(mode1, **array_kwargs)
    # _, _, Zb = get_array(mode2, **array_kwargs)

    X, Y, Za = read_ascii_array(mode1, **ascii_array_kwargs)
    _, _, Zb = read_ascii_array(mode2, **ascii_array_kwargs)

    peaks_a, peaks_b = [ get_local_peaks(z, peak_type='minimum') for z in Za, Zb ]
    peaks_a[np.logical_or(Y > 0.9, Y < 0.1)] = 0.0
    peaks_b[np.logical_or(Y > 0.9, Y < 0.1)] = 0.0
    idx_a, idx_b = [ np.where(p) for p in peaks_a, peaks_b ]

    if plot:
        f, (ax1, ax2) = plt.subplots(nrows=2, figsize=(200, 100))
        cmap = plt.cm.jet
        cmap.set_bad('dimgrey', 1)

        scale = 5.
        ax1.pcolormesh(X, Y, scale*Za, cmap=cmap)
        ax1.scatter(X[idx_b], Y[idx_b], s=1.5e4, c="w", edgecolors=None)

        ax2.pcolormesh(X, Y, scale*Zb, cmap=cmap)
        ax2.scatter(X[idx_b], Y[idx_b], s=1.5e4, c="w", edgecolors=None)

        for ax in (ax1, ax2):
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())

        plt.savefig('wavefunction_peaks_wavefunction.jpg', bbox_inches='tight')
        print "Wavefunction written."

    Zp = np.zeros_like(X)
    for (xn, yn) in zip(X[idx_b].flatten(), Y[idx_b].flatten()):
            Zp -= gauss(X, xn, sigma) * gauss(Y, yn, sigma)

    np.savetxt("wavefunction_peaks_potential.dat",
               zip(range(len(Zp.flatten('F'))), Zp.flatten('F')))
    np.savez("wavefunction_peaks_potential.npz", X=X, Y=Y, P=Zp)
    print "Potential written."


if __name__ == '__main__':
    argh.dispatch_command(main)
