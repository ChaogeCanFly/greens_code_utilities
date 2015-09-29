#!/usr/bin/env python2.7

import json
import multiprocessing
import numpy as np
import os
import matplotlib
if os.environ.get('SLURM_NTASKS'):
    matplotlib.use("Agg")
    print "Using 'Agg' backend..."
from scipy.ndimage.filters import gaussian_filter, uniform_filter
from scipy import stats
import sys

import argh

from ascii_to_numpy import read_ascii_array
from ep.helpers import get_local_peaks
from helper_functions import convert_json_to_cfg

FILE_NAME = "peaks"
PIC_ASCII_YMIN = 0.2375
PIC_ASCII_YMAX = 0.7500
POT_CUTOFF_VALUE = -1.0
PLOT_FIGSIZE = (200, 100)
# PLOT_FIGSIZE_SCALING = 250
PLOT_FIGSIZE_SCALING = 25
PLOT_FONTSIZE = 100
PICKER_TOLERANCE = 5


def on_pick(event, event_coordinates, fig):
    """Record (x, y) coordinates at each click and print to file."""
    event = event.mouseevent
    xmouse, ymouse = event.xdata, event.ydata

    ax = fig.get_axes()[0]

    if event.button == 1:
        print "x, y:", xmouse, ymouse
        event_coordinates.append([xmouse, ymouse])
        x, y = np.asarray(event_coordinates).T
        ax.scatter(x, y, s=1e2, c="k", edgecolors=None)
    elif event.button == 3:
        x, y = np.asarray(event_coordinates).T
        x_close = np.isclose(x, xmouse, rtol=1e-2)
        y_close = np.isclose(y, ymouse, rtol=1e-2)
        try:
            idx = np.where(x_close & y_close)[0][0]
            ax.scatter(x[idx], y[idx], s=1e2, c="red", marker="x")
            del event_coordinates[idx]
        except:
            print "No point found near ({}, {}).".format(xmouse, ymouse)

    fig.canvas.draw()


def on_key(event, plt):
    """Quit the interactive session based on a keypress event."""
    if event.key in 'q':
        plt.close()


@argh.arg('--mode1', type=str)
@argh.arg('--mode2', type=str)
@argh.arg('--npz-potential', type=str)
@argh.arg('--txt-potential', type=str)
@argh.arg('--write-peaks', type=str)
@argh.arg('--r-nx', type=int)
@argh.arg('--r-ny', type=int)
@argh.arg('--shift', type=str)
@argh.arg('--threshold', type=float)
@argh.arg('--cutoff', type=float)
@argh.arg('--limits', type=float, nargs='+')
@argh.arg('--eta0', type=float)
def main(pphw=50, N=2.6, L=10., W=1., sigmax=10., sigmay=1.,
         amplitude=1., r_nx=None, r_ny=None, plot=False,
         pic_ascii=False, write_peaks=None, mode1=None, mode2=None,
         npz_potential=None, txt_potential=None, peak_function='local',
         savez=False, threshold=None, shift=None, interpolate=0,
         limits=[1e-2, 0.99, 5e-2, 0.95], dryrun=False, no_mayavi=False,
         interactive=False, filter='uniform', cutoff=None, eta0=None):
    """Generate greens_code potentials from *.ascii files.

        Parameters:
        -----------
            pphw: int
                points per halfwave
            N: int
                number of open modes
            L, W: float
                system length, width
            sigmax, sigmay: float
                potential extension in x- and y-direction in % of width W
            amplitude: float
                potential amplitude
            r_nx, r_ny: int
                number of gridpoints in x- and y-direction
            plot: bool
                whether to plot wavefunctions and potentials
            pic_ascii: bool
                build potential from pic.*.ascii files
            write_peaks: int (1|2)
                whether to construct a potential from mode 1 or 2
            mode1, mode2: str
                *.ascii file of mode 1 and 2
            npz_potential: str
                if supplied, use .npz file as input
            txt_potential: str
                use peaks from external text file
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
            no_mayavi: bool
                whether to produce a 3D plot of the potential
            interactive: bool
                whether to open an interactive plot window to select indiviual
                points
            filter: str (gauss|uniform)
                chooses which filter to apply
            eta0: float
                constant absorption background
    """
    settings = json.dumps(vars(), sort_keys=True, indent=4)
    print settings + "\n"
    with open(FILE_NAME + '.json', 'w') as f:
        f.write(settings)
    convert_json_to_cfg(infile=FILE_NAME + '.json',
                        outfile=FILE_NAME + '.cfg')

    if dryrun:
        sys.exit()

    if npz_potential:
        npz_file = np.load(npz_potential)
        npz_vars = ('X', 'Y', 'Z_1', 'Z_2', 'P', 'x', 'y')
        X, Y, Z_1, Z_2, P, x, y = [npz_file[s] for s in npz_vars]
    else:
        ascii_array_kwargs = {'L': L,
                              'W': W,
                              'pphw': pphw,
                              'N': N,
                              'r_nx': r_nx,
                              'r_ny': r_ny,
                              'pic_ascii': pic_ascii,
                              'return_abs': True}
        print "Reading .ascii files..."
        try:
            pool = multiprocessing.Pool(processes=2)
            R = [pool.apply_async(read_ascii_array, args=(m,),
                                  kwds=ascii_array_kwargs) for m in (mode1,
                                                                     mode2)]
            (X, Y, Z_1), (_, _, Z_2) = [r.get() for r in R]
        except:
            X, Y, Z_1 = read_ascii_array(mode1, **ascii_array_kwargs)
            _, _, Z_2 = read_ascii_array(mode2, **ascii_array_kwargs)

    if plot or interactive:
        # import matplotlib
        from matplotlib import pyplot as plt
        from ep.plot import get_colors

        _, cmap, _ = get_colors()

    if write_peaks:
        if write_peaks == '1':
            Z = Z_1
        elif write_peaks == '2':
            Z = Z_2

        print "Building potential based on mode {}...".format(write_peaks)
        P = np.zeros_like(X)

        if len(limits) != 4:
            raise Exception("Error: --limits option needs exactly 4 entries.")
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
        elif 'cut' in peak_function:
            peaks = np.logical_and(Z < threshold*Z.max(), WG_mask)
        elif 'const' in peak_function:
            peaks = np.ones_like(Z)

        # get array-indices of peaks
        idx = np.where(peaks)
        print "Found {} peaks...".format(len(idx[0]))
        x, y = [u[idx].flatten() for u in (X, Y)]

        if txt_potential:
            print "Loading txt_potential..."
            x, y = np.loadtxt(txt_potential, unpack=True)

        if interactive:
            print "Starting interactive session..."

            fig, ax = plt.subplots()

            ax.pcolormesh(X, Y, Z, picker=PICKER_TOLERANCE, cmap=cmap)
            ax.scatter(x, y, s=5e1, c="w", edgecolors=None)
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())

            event_coordinates = []
            on_pick_lambda = lambda s: on_pick(s, event_coordinates, fig)
            key_press_lambda = lambda s: on_key(s, plt)
            fig.canvas.callbacks.connect('pick_event', on_pick_lambda)
            fig.canvas.callbacks.connect('key_press_event', key_press_lambda)
            plt.show()
            try:
                x, y = np.asarray(event_coordinates).T
            except:
                print """Warning: event_coordinates cannot be unpacked.
                Proceeding with old (x,y) values."""

        # sort coordinates wrt x-coordinate
        x, y = [u[np.argsort(x)] for u in (x, y)]

        if interpolate:
            print "Interpolating data points..."
            from scipy.interpolate import splprep, splev

            tck, _ = splprep([x, y], s=0.0, k=1)
            x, y = splev(np.linspace(0, 1, interpolate), tck)

        # reapply limits
        x_mask = (x > L*limits[0]) & (x < L*limits[1])
        x, y = [u[x_mask] for u in x, y]

        # write potential to grid-points
        print "Writing potential to grid-points..."
        # TODO: factor was 1.05 - introduces bugs?
        # eps = W/P.shape[0]*1.10
        # for xi, yi in zip(x, y):
        #     zi = np.where((np.abs(X - xi) < eps) & (np.abs(Y - yi) < eps))
        #     P[zi] = POT_CUTOFF_VALUE
        if 'const' in peak_function:
            P = 1.*peaks
        else:
            dx = L/P.shape[1]
            xn, yn = [np.floor(u/dx) for u in x, y]
            for xi, yi in zip(xn, yn):
                P[yi, xi] = POT_CUTOFF_VALUE


        # sigma here is in % of waveguide width W (r_ny) [caveat: P = P(y,x)]
        sigmax, sigmay = [P.shape[0]*s/100. for s in sigmax, sigmay]

        # decorate data points with filter
        print "Applying filter..."
        if 'uniform' in filter:
            P = uniform_filter(P, (sigmay, sigmax), mode='constant')
        elif 'gauss' in filter:
            P = gaussian_filter(P, (sigmay, sigmax), mode='constant')

        # normalize potential based on most frequent value P_ij < 0.
        print "Normalize potential..."
        if not cutoff:
            print "Determine cutoff..."
            cutoff = stats.mode(P[P < 0.])[0][0]
            print "cutoff value:", cutoff
        P[P < 0.99*cutoff] = POT_CUTOFF_VALUE
        P /= -P.min()

        if 'sine' in peak_function:
            print "Applying sine envelope..."
            L0 = L*(limits[1] - limits[0])/2.
            envelope = np.sin(np.pi/(2.*L0)*(X - L*limits[0]))
            P *= envelope

        if 'eps' in peak_function:
            print "Applying eps_sq envelope..."
            print "WARNING: using eps(x)/eps0 = 0.5(1-cos(2pi/Lx)) parametrization!"

            L0 = L*(limits[1] - limits[0])/2.
            envelope = 0.5*(1. - np.cos(np.pi/L0*(X - L*limits[0])))
            if 'sq' in peak_function:
                envelope *= envelope
            P *= envelope

        if shift:
            print "Shifting indices of target array..."
            for n, vn in np.loadtxt(shift):
                P[:, n] = np.roll(P[:, n], -int(vn), axis=0)

        # scale potential
        P *= amplitude

        if eta0:
            P += eta0

        print "Writing potential based on mode {}...".format(write_peaks)
        np.savetxt("mode_{}_peaks_potential.dat".format(write_peaks),
                   list(enumerate(P.flatten('F'))))

        # always write the potential coordinates
        print "Writing coordinates file..."
        np.savetxt(FILE_NAME + '.dat', zip(x, y))

        if savez:
            print "Writing .npz file..."
            np.savez(FILE_NAME + '.npz',
                     X=X, Y=Y, Z_1=Z_1, Z_2=Z_2, P=P, x=x, y=y)

    if plot:
        print "Plotting wavefunctions..."
        matplotlib.rcParams.update({'font.size': PLOT_FONTSIZE})

        f, (ax1, ax2) = plt.subplots(nrows=2, figsize=PLOT_FIGSIZE)

        # scattering wavefunction
        ax1.pcolormesh(X, Y, Z_1, cmap=cmap)
        ax2.pcolormesh(X, Y, Z_2, cmap=cmap)

        if write_peaks:
            ax1.scatter(x, y, s=1.5e4, c="w", edgecolors=None)
            ax2.scatter(x, y, s=1.5e4, c="w", edgecolors=None)

        # if npz_potential:
        #     X_nodes = npz_file['x']
        #     Y_nodes = npz_file['y']
        #     ax1.scatter(X_nodes, Y_nodes, s=1e4, c="k", edgecolors=None)
        #     ax2.scatter(X_nodes, Y_nodes, s=1e4, c="k", edgecolors=None)

        for ax in (ax1, ax2):
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())

        plt.savefig(FILE_NAME + '_wavefunction.png', bbox_inches='tight')

        print "Plotting 2D potential..."
        f, ax = plt.subplots(figsize=(PLOT_FIGSIZE_SCALING*L,
                                      PLOT_FIGSIZE_SCALING*W))
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.grid(True)
        p = ax.pcolormesh(X, Y, P, cmap=cmap)
        f.colorbar(p)
        ax.set_aspect('equal', 'datalim')
        plt.savefig(FILE_NAME + '_potential_2D.png', bbox_inches='tight')

        if not no_mayavi:
            try:
                print "Plotting 3D potential..."
                from mayavi import mlab

                mlab.figure(size=(1024, 756))
                extent = (0, 1, 0, 5, 0, 1)
                p = mlab.surf(-P, extent=extent)
                cmap = cmap(np.arange(256))*255.
                p.module_manager.scalar_lut_manager.lut.table = cmap
                mlab.view(distance=7.5)
                mlab.savefig(FILE_NAME + '_potential_3D.png')
            except:
                print "Error: potential image not written."


if __name__ == '__main__':
    argh.dispatch_command(main)
