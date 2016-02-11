#!/usr/bin/env python2.7

import itertools
import numpy as np
import sys

import argh

from helper_functions import convert_to_complex


def read_ascii_array(ascii_file, L=None, W=None, pphw=None, N=None, r_nx=None,
                     r_ny=None, return_abs=True, pic_ascii=False):
    """Read the wavefunction in .ascii format and return the data together with
    a XY meshgrid."""

    if not pic_ascii:
        n, Z = np.genfromtxt(ascii_file, unpack=True, usecols=(0, 1),
                             dtype=complex, converters={1: convert_to_complex})
        if r_nx is None and r_nx is None:
            if None not in (pphw, N, L, W):
                nyout = pphw*N
                dy = W/(nyout + 1.)
                r_nx = int(L/dy)
                r_ny = int(W/dy)
            else:
                raise Exception(("Error: either pphw/N or r_nx/r_ny have "
                                 "to be supplied."))

            shift = xrange(-50, 50)
            for (xn, yn) in itertools.product(shift, repeat=2):
                r_nx_shift = r_nx + xn
                r_ny_shift = r_ny + yn
                try:
                    print "Trying ({}, {})".format(r_nx_shift, r_ny_shift)
                    Z = Z.reshape(r_ny_shift, r_nx_shift, order='F')
                    print "Successfull for (xn, yn) = ({}, {})".format(xn, yn)
                    break
                except ValueError as error:
                    warning = "Warning: {}. Cannot use (r_xn, r_yn) = ({}, {})."
                    print warning.format(error, r_nx_shift, r_ny_shift)

            r_nx, r_ny = r_nx_shift, r_ny_shift
        else:
            Z = Z.reshape(r_ny, r_nx, order='F')

        x = np.linspace(0., L, r_nx)
        y = np.linspace(0., W, r_ny)
        X, Y = np.meshgrid(x, y)
    else:
        n, m, Z_re, Z_im = np.loadtxt(ascii_file, unpack=True)
        xdim = n[-1] + 1
        ydim = m[-1] + 1
        N, M, Z_RE, Z_IM = [x.reshape(xdim, ydim, order='C') for
                            x in (n, m, Z_re, Z_im)]
        X = N/N.max()*L
        Y = M/M.max()*W  # should be W + 2.*epsilon
        Z = Z_RE + 1j*Z_IM

    if return_abs:
        Z = np.abs(Z)**2

    return X, Y, Z


@argh.arg('ascii-file', type=str)
def main(ascii_file, pphw=50, N=2.5, L=100., W=1., plot=False, pic_ascii=False,
         output="ascii_to_numpy"):

    X, Y, Z = read_ascii_array(ascii_file, pphw=pphw, N=N, L=L, W=W,
                               return_abs=True, pic_ascii=pic_ascii)

    print "Writing .npz file..."
    np.savez(output + ".npz", X=X, Y=Y, Z=Z)
    print "done."

    if plot:
        from matplotlib import pyplot as plt
        print "Plotting..."
        f, ax1 = plt.subplots(nrows=1, figsize=(2*L, L/2))
        cmap = plt.cm.jet

        ax1.pcolormesh(X, Y, Z, cmap=cmap)

        for ax in (ax1, ):
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())

        plt.savefig(output + '.jpg', bbox_inches='tight')
        print "done."


if __name__ == '__main__':
    argh.dispatch_command(main)
