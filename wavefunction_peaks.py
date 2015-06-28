#!/usr/bin/env python2.7

import json
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import argh

from ascii_to_numpy import read_ascii_array
from ep.helpers import get_local_peaks, get_local_minima
from ep.potential import gauss
from helper_functions import convert_to_complex


@argh.arg('--mode1', type=str)
@argh.arg('--mode2', type=str)
@argh.arg('--potential', type=str)
@argh.arg('--write-peaks', type=str)
@argh.arg('--r-nx', type=int)
@argh.arg('--r-ny', type=int)
def main(pphw=50, N=2.5, L=100., W=1., sigma=0.01, plot=False, r_nx=None, r_ny=None,
         pic_ascii=False, write_peaks=None, mode1=None, mode2=None,
         potential=None, peak_function='local', savez=False, threshold=5e-3):

    settings = json.dumps(vars(), sort_keys=True, indent=4)
    print settings
    with open("potential.cfg", "w") as f:
        f.write(settings)

    print "\nReading .ascii files..."
    ascii_array_kwargs = {'L': L,
                          'W': W,
                          'pphw': pphw,
                          'N': N,
                          'r_nx': r_nx,
                          'r_ny': r_ny,
                          'pic_ascii': pic_ascii,
                          'return_abs': True}
    X, Y, Z_1 = read_ascii_array(mode1, **ascii_array_kwargs)
    _, _, Z_2 = read_ascii_array(mode2, **ascii_array_kwargs)
    print "done."

    if potential:
        P_npz = np.load(potential)
        P = P_npz['P']
        # transform maxima to minima
        P = P.max() - P

    if write_peaks:
        if write_peaks == '1':
            Z = Z_1
        elif write_peaks == '2':
            Z = Z_2

        print "Building potential based on mode {}...".format(write_peaks)
        Z_pot = np.zeros_like(X)

        X_mask = np.logical_and(0.01*L < X, X < 0.99*L)
        Y_mask = np.logical_and(0.05*W < Y, Y < 0.95*W)
        WG_mask = np.logical_and(X_mask, Y_mask)
        sigma /= 100.  # sigma in %

        if peak_function == 'local':
            peaks = get_local_peaks(Z, peak_type='minimum')
            # remove minma due to boundary conditions at walls
            peaks[~Y_mask] = 0.0

        elif peak_function == 'cut':
            peaks = np.logical_and(Z < threshold*Z.max(), WG_mask)

        elif peak_function == 'points':
            peaks = np.logical_and(Z < threshold*Z.max(), WG_mask)
            Z_pot[np.where(peaks)] = -1.0
            # sigma here is in % of waveguide width W (r_ny)
            sigma = Z_pot.shape[0]*sigma  # caveat: Z_pot = Z_pot(y,x)
            Z_pot = gaussian_filter(Z_pot, sigma, mode='constant')
            Z_pot[Z_pot < -0.1] = -0.1
            Z_pot /= -Z_pot.min()  # normalize potential

        elif peak_function == 'fermi':
            peaks = np.logical_and(Z < threshold*Z.max(), WG_mask)
            Z_pot[np.where(peaks)] = -1.0
            # sigma here is in % of waveguide width W (r_ny)
            sigma = Z_pot.shape[0]*sigma  # caveat: Z_pot = Z_pot(y,x)
            Z_pot = gaussian_filter(Z_pot, sigma, mode='constant')
            Z_pot[Z_pot < -0.1] = -0.1
            Z_pot /= -Z_pot.min()  # normalize potential

            def fermi(x, sigma):
                return 1./(1. + np.exp(-x/sigma))

            s = 0.500
            Z_pot = fermi(Z_pot-4.*s, s)*fermi(L-Z_pot-4.*s, s)


        # get array-indices of peaks
        idx = np.where(peaks)
        print "...found {} peaks...".format(len(idx[0]))

        if peak_function == 'local' or peak_function == 'cut':
            # build Gaussian potential at peaks
            sigma *= W  # scale sigma with waveguide dimensions
            for n, (xn, yn) in enumerate(zip(X[idx].flatten(),
                                             Y[idx].flatten())):
                if n % 100 == 0:
                    print "iteration step n=", n
                Z_pot -= (np.exp(-0.5*((X-xn)**2+(Y-yn)**2)/sigma**2)/
                            (2.*np.pi*sigma**2))
            print "done."

        print "Writing potential based on mode {}...".format(write_peaks)
        np.savetxt("mode_{}_peaks_potential.dat".format(write_peaks),
                   zip(range(len(Z_pot.flatten('F'))), Z_pot.flatten('F')))
        if savez:
            np.savez("mode_{}_peaks_potential.npz".format(write_peaks),
                     X=X, Y=Y, Z_1=Z_1, Z_2=Z_2, P=Z_pot,
                     X_nodes=X[idx], Y_nodes=Y[idx])
        print "done."

    if plot:
        print "Plotting wavefunctions..."
        from matplotlib import pyplot as plt
        from ep.plot import get_colors

        f, (ax1, ax2) = plt.subplots(nrows=2, figsize=(200, 100))
        get_colors()
        cmap = plt.cm.get_cmap('parula')

        # scattering wavefunction
        ax1.pcolormesh(X, Y, Z_1, cmap=cmap)
        ax2.pcolormesh(X, Y, Z_2, cmap=cmap)

        if write_peaks:
            ax1.scatter(X[idx], Y[idx], s=1.5e4, c="w", edgecolors=None)
            ax2.scatter(X[idx], Y[idx], s=1.5e4, c="w", edgecolors=None)

        if potential:
            X_nodes = P_npz['X_nodes']
            Y_nodes = P_npz['Y_nodes']
            ax1.scatter(X_nodes, Y_nodes, s=1e4, c="k", edgecolors=None)
            ax2.scatter(X_nodes, Y_nodes, s=1e4, c="k", edgecolors=None)

        for ax in (ax1, ax2):
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())

        plt.savefig('wavefunction.png', bbox_inches='tight')
        if savez:
            np.savez('wavefunction.npz', X=X, Y=Y, Z_1=Z_1, Z_2=Z_2)
        print "done."

        print "Plotting potential..."
        try:
            from mayavi import mlab
            extent = (0, 1, 0, 5, 0, 1)
            p = mlab.surf(-Z_pot, extent=extent)
            p.module_manager.scalar_lut_manager.lut.table = cmap(np.arange(256))*255.
            mlab.savefig('potential.png')
        except:
            print "Error: potential.png not written."
        print "done."


if __name__ == '__main__':
    argh.dispatch_command(main)
