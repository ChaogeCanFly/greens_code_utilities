#!/usr/bin/env python2.7

from glob import glob
import numpy as np
from matplotlib import pyplot as plt

import argh

from ep.helpers import get_local_peaks, get_local_minima
from ep.potential import gauss
from get_wavefunction_peaks import get_array
from helper_functions import convert_to_complex


@argh.arg('mode1', type=str)
@argh.arg('mode2', type=str)
@argh.arg('potential', type=str)
def main(pphw=50, N=2.5, L=100, W=1, eps=0.1, sigma=0.01, plot=True,
         mode1=None, mode2=None, potential=None):
    nyout = pphw*N + 1.
    r_nx_pot = int(nyout*L)
    r_ny_pot = int(nyout*W)
    print "r_nx_pot, r_ny_pot", r_nx_pot, r_ny_pot

    array_kwargs = {'L': L,
                    'W': W,
                    'r_nx': r_nx_pot,
                    'r_ny': r_ny_pot}

    if mode1 is None:
        mode1 = glob("*.0000.streu.*.purewavefunction.ascii")[0]
    if mode2 is None:
        mode2 = glob("*.0001.streu.*.purewavefunction.ascii")[0]
    if potential is None:
        potential = glob("potential.*.purewavefunction.ascii")[0]

    X, Y, Za = get_array(mode1, **array_kwargs)
    _, _, Zb = get_array(mode2, **array_kwargs)
    _, _, P = get_array(potential, **array_kwargs)
    P = P.max() - P

    np.save("potential.npy", P)
    np.save("potential_x.npy", X)
    np.save("potential_y.npy", Y)
    print "Potential written."

    peaks_a, peaks_b, peaks_p = [ get_local_peaks(z, peak_type='minimum') for z in Za, Zb, P ]
    print "peaks_a.shape", peaks_a.shape
    print "peaks_b.shape", peaks_b.shape
    print "peaks_p.shape", peaks_p.shape
    peaks_a[np.logical_or(Y > 0.9, Y < 0.1)] = 0.0
    peaks_b[np.logical_or(Y > 0.9, Y < 0.1)] = 0.0
    idx_a, idx_b, idx_p = [ np.where(p) for p in peaks_a, peaks_b, peaks_p ]

    if plot:
        f, (ax1, ax2) = plt.subplots(nrows=2, figsize=(200, 100))
        cmap = plt.cm.jet
        # cmap.set_bad('dimgrey', 1)

        ax1.pcolormesh(X, Y, Za, cmap=cmap)
        # ax1.scatter(X[idx_b], Y[idx_b], s=1.5e4, c="w", edgecolors=None)
        mask = P < P.max()*6e-1
        # mask = idx_p
        ax1.scatter(X[mask], Y[mask], s=1.5e4, c="w", edgecolors=None)
        # ax1.scatter(X[mask], Y[mask], s=1e4, c="k", edgecolors=None)

        ax2.pcolormesh(X, Y, Zb, cmap=cmap)
        # ax2.scatter(X[idx_b], Y[idx_b], s=1.5e4, c="w", edgecolors=None)
        ax2.scatter(X[mask], Y[mask], s=1.5e4, c="w", edgecolors=None)
        # ax2.scatter(X[mask], Y[mask], s=1e4, c="k", edgecolors=None)

        for ax in (ax1, ax2):
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())

        plt.savefig('wavefunction.jpg', bbox_inches='tight')
        print "Wavefunction written."


if __name__ == '__main__':
    argh.dispatch_command(main)
