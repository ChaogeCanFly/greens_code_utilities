#!/usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt
try:
    import mayavi.mlab as mlab
except ImportError:
    print "Warning: mayavi.mlab not found!"
import sys

import argh

from helper_functions import unique_array


def reorder_file(infile="bloch.tmp"):
    """Reorder a file containing a function on a shuffled meshgrid."""

    eps, delta, ev0r, ev0i, ev1r, ev1i = np.loadtxt(infile, unpack=True)

    ev0 = ev0r+1j*ev0i
    ev1 = ev1r+1j*ev1i

    _, idx = unique_array(np.array(zip(eps, delta)))
    eps, delta, ev0, ev1 = [ x[idx] for x in eps, delta, ev0, ev1 ]

    idx = np.lexsort((delta, eps))
    eps, delta, ev0, ev1 = [ x[idx] for x in eps, delta, ev0, ev1 ]

    return eps, delta, ev0, ev1


def find_outliers(eps):
    """Find rows with missing values and return their index."""
    error_values = []

    for e in np.unique(eps):
        e_count = np.count_nonzero(e == eps)
        error_values.append(e_count)

    e_max = max(error_values)

    mask = np.ones_like(eps, dtype=bool)
    for e in np.unique(eps):
        e_count = np.count_nonzero(e == eps)
        if e_count < e_max:
            print "{:.6f}".format(e), e_count
            mask[np.where(e == eps)] = False

    return mask


@argh.arg("-p", "--png", type=str)
@argh.arg("-l", "--limits", type=float, nargs="+")
@argh.arg("-o", "--outfile", type=str)
@argh.arg("-t", "--trajectory", type=str)
@argh.arg("-i", "--infile")
@argh.arg("-I", "--interpolate")
def plot_3D_spectrum(infile="bloch.tmp", outfile=None, trajectory=None,
                     reorder=False, jump=100., mayavi=False, limits=None,
                     sort=False, png=None, full=False, dryrun=False, 
                     interpolate=False):
    """Visualize the eigenvalue spectrum with mayavi.mlab's mesh (3D) and
    matplotlib's pcolormesh (2D).

        Parameters:
        -----------
            infile: str
                Input file.
            outfile: str
                Whether to save the reordered and sorted array before plotting.
            reorder: bool
                Whether to properly sort the input array.
            jump: float
                Whether to remove jump in the eigenvalue surface that exceed
                a given value.
            mayavi: bool
                Whether to produce 3D plots. If false, heatmaps are plotted.
            limits: list
                Set the x- and ylim: [xmin, xmax, ymin, ymax]
            sort: bool
                Sorts the eigenvalues such that one is larger than the other.
            png: str
                Save the heatmap plots in a .png file.
            full: bool
                Add additional heatmap plots.
            trajectory: str
                Plot a trajectory on top of the heatmap.
            dryrun: bool
                Whether to only return the approximate EP position.
            interpolate: bool
                Whether to interpolate |ev0-ev1| before determining the EP position.
    """

    eps, delta, ev0r, ev0i, ev1r, ev1i = np.loadtxt(infile).T
    ev0 = ev0r + 1j*ev0i
    ev1 = ev1r + 1j*ev1i
    # workound if precision changes for multiple runs
    delta = np.around(delta, decimals=8)
    len_eps, len_delta = [ len(np.unique(x)) for x in eps, delta ]

    if reorder:
        print "reordering..."
        eps, delta, ev0, ev1 = reorder_file(infile)

    if sort:
        tmp0, tmp1 = 1.*ev0, 1.*ev1
        tmp0[ev1 > ev0] = ev1[ev1 > ev0]
        tmp1[ev1 > ev0] = ev0[ev1 > ev0]

        ev0, ev1 = 1.*tmp0, 1.*tmp1

    # get eps/delta meshgrid
    try:
        eps, delta, ev0, ev1 = [ x.reshape(len_eps, len_delta) for
                                                    x in eps, delta, ev0, ev1 ]
    except ValueError as e:
        print "Data matrix has missing values. Removing outliers:"
        mask = find_outliers(eps)
        len_masked_eps = len(np.unique(eps[mask]))
        eps, delta, ev0, ev1 = [ x[mask].reshape(len_masked_eps, -1) for
                                                    x in eps, delta, ev0, ev1 ]

    # set x/y limits
    if limits:
        eps_min, eps_max = limits[:2]
        delta_min, delta_max = limits[2:]
        limit_mask = ((eps > eps_min) & (eps < eps_max) &
                      (delta > delta_min) & (delta < delta_max))
        for X in ev0, ev1:
            X[~limit_mask] = np.nan

    if outfile:
        v = (eps, delta, ev0.real, ev0.imag, ev1.real, ev1.imag)
        v = np.array([ x.flatten() for x in v ])
        np.savetxt(outfile, v.T, fmt='%.18f')
        sys.exit()

    # remove Nan values
    ev0 = np.ma.masked_where(np.isnan(ev0), ev0)
    ev1 = np.ma.masked_where(np.isnan(ev1), ev1)

    # print minimum of eigenvalue difference
    i, j = np.unravel_index(np.sqrt((ev0.real-ev1.real)**2 +
                                    (ev0.imag-ev1.imag)**2).argmin(), ev0.shape)
    print "Approximate EP location:"
    print "eps_EP =", eps[i,j]
    print "delta_EP =", delta[i,j]

    if dryrun and not interpolate:
        sys.exit()

    if mayavi:
        extent = (0, 1, 0, 1, 0, 1)
        # real part
        for e in ev0, ev1:
            mask = np.zeros_like(eps).astype(bool)
            mask[np.abs(np.diff(e.real, axis=0)) > jump] = True
            mask[np.abs(np.diff(e.real, axis=1)) > jump] = True

            try:
                mask = np.logical_or(mask, ~limit_mask)
            except:
                pass
            mlab.figure(0, bgcolor=(0.5,0.5,0.5))
            m1 = mlab.mesh(eps.real, delta.real, e.real, mask=mask,
                           extent=extent)
            # m1.actor.actor.scale = (5,1,1)

        mlab.title("Real part", opacity=0.25)
        mlab.axes(color=(0,0,0), nb_labels=3, xlabel="epsilon", ylabel="delta",
                  zlabel="Re(K)")

        # imag part
        for e in ev0, ev1:
            mask = np.zeros_like(eps).astype(bool)
            mask[np.abs(np.diff(e.imag, axis=0)) > jump] = True
            mask[np.abs(np.diff(e.imag, axis=1)) > jump] = True

            mlab.figure(1, bgcolor=(0.5,0.5,0.5))
            try:
                mask = np.logical_or(mask, ~limit_mask)
            except:
                pass
            m2 = mlab.mesh(eps.real, delta.real, e.imag, mask=mask,
                           extent=extent)
            # m2.actor.actor.scale = (5,1,1)

        mlab.title("Imaginary part", opacity=0.25)
        mlab.axes(color=(0,0,0), nb_labels=3, xlabel="epsilon", ylabel="delta",
                  zlabel="Im(K)")

        mlab.show()

    elif full:
        f, axes = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True)
        (ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8) = axes

        plt.xticks(rotation=70)
        plt.suptitle(infile)
        cmap = plt.get_cmap('Blues_r')

        ax1.set_title(r"$\Re K_0$")
        im1 = ax1.pcolormesh(eps, delta, ev0.real, cmap=cmap)
        ax2.set_title(r"$\Im K_0$")
        im2 = ax2.pcolormesh(eps, delta, ev0.imag, cmap=cmap)
        ax3.set_title(r"$\Re K_1$")
        im3 = ax3.pcolormesh(eps, delta, ev1.real, cmap=cmap)
        ax4.set_title(r"$\Im K_1$")
        im4 = ax4.pcolormesh(eps, delta, ev1.imag, cmap=cmap)

        ax5.set_title(r"$|\Re K_0 - \Re K_1|^2$")
        Z_real = abs(ev1.real - ev0.real)
        im5 = ax5.pcolormesh(eps, delta, Z_real, cmap=cmap, vmin=0)
        ax6.set_title(r"$|\Im K_0 - \Im K_1|^2$")
        Z_imag = abs(ev1.imag - ev0.imag)
        im6 = ax6.pcolormesh(eps, delta, Z_imag, cmap=cmap, vmin=0)

        Z = np.sqrt(Z_imag**2 + Z_real**2)
        ax7.set_title(r"$\sqrt{(\Re K_0 - \Re K_1)^2 + (\Im K_0 - \Im K_1)^2}$")
        im7 = ax7.pcolormesh(eps, delta, Z, cmap=cmap, vmin=0)

        Z = (Z_imag**2 + Z_real**2)**0.25
        ax8.set_title(r"$\sqrt[4]{(\Re K_0 - \Re K_1)^2 + (\Im K_0 - \Im K_1)^2}$")
        im8 = ax8.pcolormesh(eps, delta, Z, cmap=cmap, vmin=0)

        for im, ax in zip((im1, im2, im3, im4, im5, im6, im7, im8),
                          (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8)):
            ax.set_xlabel("epsilon")
            ax.set_ylabel("delta")
            ax.set_xlim(eps.min(), eps.max())
            ax.set_ylim(delta.min(), delta.max())
            f.colorbar(im, ax=ax)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        if png:
            plt.savefig(png)
        else:
            plt.show()
    else:
        f, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)

        plt.xticks(rotation=90)
        plt.suptitle(infile)

        Z_real = abs(ev1.real - ev0.real)
        Z_imag = abs(ev1.imag - ev0.imag)
        Z = np.sqrt(Z_imag**2 + Z_real**2)

        ax.set_title(r"$\sqrt{(\Re K_0 - \Re K_1)^2 + (\Im K_0 - \Im K_1)^2}$")

        # add one column and row to meshgrids such that meshgrid doesn't cut 
        # away any important data
        eps_u = np.unique(eps)
        delta_u = np.unique(delta)
        eps0, delta0 = np.meshgrid(np.concatenate((eps_u, [2*eps.max()])),
                                   np.concatenate((delta_u, [2*delta.max()])))
        Z0 = np.c_[Z, Z[:,-1]]
        Z0 = np.vstack((Z0, Z0[-1]))

        if interpolate:
            from scipy.interpolate import griddata
            x = np.linspace(eps.min(), eps.max(), 500)
            y = np.linspace(delta.min(), delta.max(), 1000)
            X, Y = np.meshgrid(x,y)
            Z0 = griddata((eps.ravel(), delta.ravel()), Z.ravel(), (X, Y), method='cubic')
            i, j = np.unravel_index(Z0.argmin(), Z0.shape)
            print "eps_EP_interpolated =", X[i,j]
            print "delta_EP_interpolated =", Y[i,j]
            eps0, delta0 = X.T, Y.T

            if dryrun:
                sys.exit()

        # im = ax.pcolormesh(eps0.T, delta0.T, np.log(Z0),
        im = ax.pcolormesh(eps0.T, delta0.T, np.log(Z0),
                           cmap=plt.get_cmap('Blues_r'))

        # correct ticks
        xoffset = np.diff(eps_u).mean()/2
        yoffset = np.diff(delta_u).mean()/2
        ax.set_xticks(eps_u + xoffset)
        ax.set_yticks(delta_u + yoffset)

        # ticklabels
        # ax.set_xticklabels(np.around(eps_u, decimals=4))
        # ax.set_yticklabels(np.around(delta_u, decimals=4))

        # axis labels
        ax.set_xlabel("epsilon")
        ax.set_ylabel("delta")
        if limits:
            print limits
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])
        else:
            ax.set_xlim(eps.min(), eps.max() + 2*xoffset)
            ax.set_ylim(delta.min(), delta.max() + 2*yoffset)

        f.colorbar(im, ax=ax)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.subplots_adjust(top=0.875)

        if trajectory:
            # n, eps, delta = np.loadtxt(trajectory, unpack=True)[:3]
            eps, delta = np.loadtxt(trajectory, unpack=True)[:2]
            plt.plot(eps, delta, "r-")
            # for n, (eps, delta) in enumerate(zip(eps, delta)):
            #     plt.text(eps, delta, str(n), fontsize=12)

        if png:
            plt.savefig(png)
        else:
            plt.show()


if __name__ == '__main__':
    argh.dispatch_command(plot_3D_spectrum)
