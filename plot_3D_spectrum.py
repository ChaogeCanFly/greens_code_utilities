#!/usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt
try:
    import mayavi.mlab as mlab
except:
    print "Warning: mayavi.mlab not found!"
import sys

import argh


def unique_array(a):
    """Remove duplicate entries in an array."""
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    b = unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
    return b


def reorder(infile="bloch.tmp", outfile="bloch_sorted.tmp"):
    """Reorder a file containing a function  on a shuffled meshgrid."""

    eps, delta, ev1r, ev1i, ev2r, ev2i = np.loadtxt(infile, unpack=True)

    ev1 = ev1r+1j*ev1i
    ev2 = ev2r+1j*ev2i

    len_eps = len(np.unique(eps))
    len_delta = len(np.unique(delta))
    print len_eps
    print len_delta

    ind = np.lexsort((delta, eps))
    eps, delta, ev1, ev2 = [ x[ind] for x in eps, delta, ev1, ev2 ]
    v = zip(eps, delta, ev1.real, ev1.imag, ev2.real, ev2.imag)
    v = np.array(v)
    # v = unique_array(v)

    len_eps = len(np.unique(eps))
    len_delta = len(np.unique(delta))
    print "len(eps)", len(np.unique(eps))
    print "len(delta)", len(np.unique(delta))

    np.savetxt(outfile, v)


def plot_3D_spectrum(infile="bloch.tmp", outfile="bloch_reordered.tmp",
                     reorder=False, jumps=100., mayavi=False, lim_mask=False, 
                     girtsch=False):
    """Visualize the eigenvalue spectrum with mayavi.mlab's mesh (3D) and
    matplotlib's pcolormesh (2D).

        Parameters:
        -----------
            infile: str
                Input file.
            outfile: str
                Output file for reordered arrays.
            reorder: bool
                Whether to properly sort the input array.
            jumps: float
                Whether to remove jumps in the eigenvalue surface that exceed
                a given value.
            mayavi: bool
                Whether to produce 3D plots. If false, heatmaps are plotted.
            lim_mask: bool
                Whether to set x- and y-ranges manually.
            girtsch: bool
                Whether to account for a wrong eps <-> delta labeling.
    """
    if reorder:
        print "reordering..."
        reorder(infile, outfile)
        sys.exit()

    if girtsch:
        eps, delta, ev0, ev1 = np.loadtxt(infile, dtype=complex).T
        # wrong labeling!
        len_eps, len_delta = [ len(np.unique(x)) for x in eps, delta ]
    else:
        eps, delta, ev0r, ev0i, ev1r, ev1i = np.loadtxt(infile).T
        ev0 = ev0r + 1j*ev0i
        ev1 = ev1r + 1j*ev1i
        len_eps, len_delta = [ len(np.unique(x)) for x in eps, delta ]

    if reorder:
        ind = np.lexsort((delta, eps))
        eps, delta, ev0, ev1 = [ x[ind] for x in eps, delta, ev0, ev1 ]

    # get eps/delta meshgrid
    eps, delta, ev0, ev1 = [ x.reshape(len_eps, len_delta) for x in
                                                        eps, delta, ev0, ev1 ]

    # remove Nan values
    ev0 = np.ma.masked_where(np.isnan(ev0), ev0)
    ev1 = np.ma.masked_where(np.isnan(ev1), ev1)

    # set x/y limits
    if lim_mask:
        mask = (eps > 0.01) & (eps < 0.09) & (delta > 0.05) & (delta < 0.5)
        for X in eps, delta, ev0, ev1:
            X[~mask] = np.nan

    # print maximum of eigenvalue
    i, j = np.unravel_index(e.imag.argmax(), ev0.shape)
    print "eps_max =", eps[i,j]
    print "delta_max =", delta[i,j]

    if mayavi:
        # real part
        mask = np.zeros_like(eps).astype(bool)
        for e in ev0, ev1:
            if not jumps:
                mask[np.abs(np.diff(ev1.real, axis=0)) > 0.07] = True
                mask[np.abs(np.diff(ev1.real, axis=1)) > 0.07] = True

            fig = mlab.figure(0, bgcolor=(0.5,0.5,0.5))
            m = mlab.mesh(eps.real, delta.real, e.real, mask=mask)
            m.actor.actor.scale = (1,1,1)

        mlab.title("Real part", opacity=0.25)
        mlab.axes(color=(0,0,0), nb_labels=3, xlabel="epsilon", ylabel="delta",
                  zlabel="Re(K)")

        # imag part
        fig = mlab.figure(1, bgcolor=(0.5,0.5,0.5))
        for e in ev0, ev1:
            mlab.mesh(eps.real, delta.real, e.imag)
            m.actor.actor.scale = (1,1,1)
        mlab.title("Imaginary part", opacity=0.25)
        mlab.axes(color=(0,0,0), nb_labels=3, xlabel="epsilon", ylabel="delta",
                  zlabel="Im(K)")
        mlab.show()
    else:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=2, ncols=2)
        plt.title(infile)
        ax1.pcolormesh(eps, delta, ev0.imag)
        ax2.pcolormesh(eps, delta, ev1.imag)
        ax3.pcolormesh(eps, delta, ev0.real)
        ax4.pcolormesh(eps, delta, ev1.real)
        plt.colorbar()
        plt.xlim(eps.min(), eps.max())
        plt.ylim(delta.min(), delta.max())
        plt.show()


if __name__ == '__main__':
    argh.dispatch_command(plot_mesh)
