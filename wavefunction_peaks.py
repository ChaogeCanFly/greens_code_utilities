#!/usr/bin/env python2.7

import json
# import multiprocessing
import numpy as np
from scipy.ndimage.filters import gaussian_filter, uniform_filter
import sys

import argh

from ascii_to_numpy import read_ascii_array
from ep.helpers import get_local_peaks
from helper_functions import convert_json_to_cfg

FILE_NAME = "wavefunction_peaks"
PIC_ASCII_YMIN = 0.2375
PIC_ASCII_YMAX = 0.7500
POT_MIN_CUTOFF = -0.05
POT_CUTOFF_VALUE = -1.0
INTERPOLATE_XY_EPS = 1e-3
PLOT_FIGSIZE = (200, 100)


@argh.arg('--mode1', type=str)
@argh.arg('--mode2', type=str)
@argh.arg('--potential', type=str)
@argh.arg('--txt-potential', type=str)
@argh.arg('--write-peaks', type=str)
@argh.arg('--r-nx', type=int)
@argh.arg('--r-ny', type=int)
@argh.arg('--shift', type=str)
@argh.arg('--limits', type=float, nargs='+')
def main(pphw=50, N=2.5, L=100., W=1., sigmax=10., sigmay=1.,
         amplitude=1., r_nx=None, r_ny=None, plot=False,
         pic_ascii=False, write_peaks=None, mode1=None, mode2=None,
         potential=None, txt_potential=None, peak_function='local',
         savez=False, threshold=5e-3, shift=None, interpolate=0,
         limits=[1e-2, 0.99, 5e-2, 0.95], dryrun=False):
    """Generate greens_code potentials from *.ascii files.

        Parameters:
        -----------
            pphw: int
            N: int
            L: float
            W: float
            sigmax: float
            sigmay: float
            amplitude: float
                potential amplitude
            r_nx: int
            r_ny: int
            plot: bool
            pic_ascii: bool
                build potential from pic.*.ascii files
            write_peaks: int (1|2)
                whether to construct a potential from mode 1 or 2
            mode1: str
                *.ascii file of mode 1
            mode2: str
                *.ascii file of mode 2
            potential: str
                if supplied, use as input
            txt_potential: str
                use peaks from external file
            peak_function: str
                determines how the potential is constructed from the
                wavefunction intensity
            savez: bool
                whether to save the output arrays in .npz format
            threshold: float
                use values < threshold*max(|psi|^2) to construct the potential
            shift: str
                use lower.dat to shift the mesh indices such that the potential
                is not distorted
            interpolate: int
                if > 0, interpolate the peaks data with n points to obtain a
                smooth potential landscape
            limits: list of floats
                determines the X- and Y-masks in percent:
                    x in [X*limits[0], X*limits[1]]
                    y in [Y*limits[2], Y*limits[4]]
            dryrun: bool
                write settings files and exit
    """
    settings = json.dumps(vars(), sort_keys=True, indent=4)
    print settings
    with open(FILE_NAME + '.json', 'w') as f:
        f.write(settings)
    convert_json_to_cfg(infile=FILE_NAME + '.json',
                        outfile=FILE_NAME + '.cfg')

    if dryrun:
        sys.exit()

    ascii_array_kwargs = {'L': L,
                          'W': W,
                          'pphw': pphw,
                          'N': N,
                          'r_nx': r_nx,
                          'r_ny': r_ny,
                          'pic_ascii': pic_ascii,
                          'return_abs': True}
    print "\nReading .ascii files..."
    X, Y, Z_1 = read_ascii_array(mode1, **ascii_array_kwargs)
    _, _, Z_2 = read_ascii_array(mode2, **ascii_array_kwargs)
    # pool = multiprocessing.Pool(processes=2)
    # R = [pool.apply_async(read_ascii_array, args=(m,),
    #                       kwds=ascii_array_kwargs) for m in (mode1, mode2)]
    # (X, Y, Z_1), (_, _, Z_2) = [r.get() for r in R]
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
        P = np.zeros_like(X)

        if len(limits) != 4:
            raise Exception("Error: len(limits) != 4.")
        # define waveguide geometry (avoid minima due to boundary conditions
        # at walls)
        X_mask = np.logical_and(limits[0]*L < X, X < limits[1]*L)
        Y_mask = np.logical_and(limits[2]*W < Y, Y < limits[3]*W)
        if pic_ascii:
            Y_mask = np.logical_and(PIC_ASCII_YMIN*W < Y, Y < PIC_ASCII_YMAX*W)
        WG_mask = np.logical_and(X_mask, Y_mask)

        if 'local' in peak_function:
            peaks = get_local_peaks(Z, peak_type='minimum')
            peaks[~WG_mask] = 0.0

        elif 'points' in peak_function:
            peaks = np.logical_and(Z < threshold*Z.max(), WG_mask)

        # get array-indices of peaks
        idx = np.where(peaks)
        print "...found {} peaks...".format(len(idx[0]))

        x, y = [u[idx].flatten() for u in (X, Y)]
        x, y = [u[np.argsort(x)] for u in (x, y)]
        if txt_potential:
            x, y = np.loadtxt(txt_potential, unpack=True)

        if interpolate:
            print "Interpolating data points..."
            from scipy.interpolate import interp1d

            if txt_potential:
                x, y = np.loadtxt(txt_potential, unpack=True)

            f = interp1d(x, y, kind='linear')
            x = np.linspace(x.min(), x.max(), interpolate)
            y = f(x)
            print "done."

        if not txt_potential:
            np.savetxt(FILE_NAME + '.dat', zip(x, y))

        # write potential on grid-points
        for xi, yi in zip(x, y):
            zi = np.where(np.logical_and(abs(X - xi) < INTERPOLATE_XY_EPS,
                                         abs(Y - yi) < INTERPOLATE_XY_EPS))
            P[zi] = POT_CUTOFF_VALUE

        # sigma here is in % of waveguide width W (r_ny) [caveat: P = P(y,x)]
        sigmax, sigmay = [P.shape[0]*s/100. for s in sigmax, sigmay]

        # decorate data points with filter
        # P = uniform_filter(P, (sigmay, sigmax), mode='constant')
        P = gaussian_filter(P, (sigmay, sigmax), mode='constant')

        # normalize potential
        P[P < POT_MIN_CUTOFF] = POT_CUTOFF_VALUE
        P /= -P.min()

        if 'sine_truncated' in peak_function:
            P /= abs(P).max()
            Xp = 1.*X
            X0 = L/4.
            Xp[np.sin(np.pi*2.*(X-X0)/L) < 0.] = X0
            f = np.sin(np.pi*2*(Xp-X0)/L)
            P *= f/f.max()
        elif 'sine' in peak_function:
            P /= abs(P).max()
            P *= np.sin(np.pi*X/L)

        if shift:
            print "Shifting indices of target array..."
            _, v = np.loadtxt(shift, unpack=True)
            for i, vi in enumerate(v):
                P[:, i] = np.roll(P[:, i], -int(vi), axis=0)
            print "done."

        # scale potential
        P *= amplitude

        print "Writing potential based on mode {}...".format(write_peaks)
        np.savetxt("mode_{}_peaks_potential.dat".format(write_peaks),
                   # zip(range(len(P.flatten('F'))), P.flatten('F')))
                   list(enumerate(P.flatten('F'))))
        print "done."
        if savez:
            print "Writing .npz file based on mode {}...".format(write_peaks)
            np.savez("mode_{}_peaks_potential.npz".format(write_peaks),
                     X=X, Y=Y, Z_1=Z_1, Z_2=Z_2, P=P,
                     X_nodes=X[idx], Y_nodes=Y[idx])
        print "done."

    if plot:
        print "Plotting wavefunctions..."
        from matplotlib import pyplot as plt
        from ep.plot import get_colors

        f, (ax1, ax2) = plt.subplots(nrows=2, figsize=PLOT_FIGSIZE)
        get_colors()
        cmap = plt.cm.get_cmap('parula')

        # scattering wavefunction
        ax1.pcolormesh(X, Y, Z_1, cmap=cmap)
        ax2.pcolormesh(X, Y, Z_2, cmap=cmap)

        if write_peaks:
            try:
                ax1.scatter(x, y, s=1.5e4, c="w", edgecolors=None)
                ax2.scatter(x, y, s=1.5e4, c="w", edgecolors=None)
            except:
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

        plt.savefig(FILE_NAME + '.png', bbox_inches='tight')
        if savez:
            np.savez(FILE_NAME + '.npz', X=X, Y=Y, Z_1=Z_1, Z_2=Z_2)
        print "done."

        print "Plotting potential..."
        try:
            from mayavi import mlab

            mlab.figure(size=(1024, 756))
            extent = (0, 1, 0, 5, 0, 1)
            p = mlab.surf(-P, extent=extent)
            cmap = cmap(np.arange(256))*255.
            p.module_manager.scalar_lut_manager.lut.table = cmap
            mlab.view(distance=7.5)
            mlab.savefig(FILE_NAME + '_potential_3D.png')

            f = plt.figure()
            plt.pcolormesh(X, Y, P)
            plt.xlim(X.min(), X.max())
            plt.ylim(Y.min(), Y.max())
            plt.savefig(FILE_NAME + '_potential_2D.png')
        except:
            print "Error: potential image not written."
        print "done."


if __name__ == '__main__':
    argh.dispatch_command(main)
