#!/usr/bin/env python

from __future__ import division

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import argh

from ep.plot import get_colors, get_defaults

colors, _, _ = get_colors()
get_defaults()


def plot_transmission(S, ax=None, swap=None):
    # N, T11, T12, T21, T22 = np.loadtxt("S_matrix.dat", unpack=True,
    #                                    usecols=(0,5,6,7,8))

    # print S_matrix.shape
    N, T11, T12, T21, T22 = S.T[[0,5,6,7,8]]
    # T12, T21 = T21, T12

    # N to frequency in GHz: N*pi/W*c/(2*pi)/10**-9
    N = N*3.

    ax.semilogy(N, T11, "-", c=colors[1], lw=2.5, mec=colors[1], label=r'$T_{11}$')
    ax.semilogy(N, T22, "-", c=colors[0], lw=2.5, mec=colors[0], label=r'$T_{22}$')
    ax.semilogy(N, T21, "-", c=colors[-1], lw=2.5, mec=colors[-1], label=r'$T_{12}$')
    ax.semilogy(N, T12, "-", c=colors[2], lw=2.5, mec=colors[2], label=r'$T_{21}$')
    ax.set_xlabel(r"$\nu$ in GHz")
    ax.set_ylabel(r"$T_{nm}$", labelpad=10)

    idx = np.where(np.isclose(N, 2.6*3, rtol=1e-2))
    # idx = np.where(np.isclose(N, 7.635, rtol=1e-3))
    print N[idx]
    print "T21 value"
    print T12[idx].mean()
    print "T21/Tij"
    for t in T11, T22, T21:
        print (T12[idx]/t[idx]).mean()

    print "T22/T11", T22[idx].mean()/T11[idx].mean()
    print "T11/T12", T11[idx].mean()/T21[idx].mean()

    return N, T12


def plot_reflection(S, T12=None, ax=None, swap=None):
    # N, R11, R12, R21, R22, Rp11, Rp12, Rp21, Rp22 = np.loadtxt("S_matrix.dat",
    #                                                            unpack=True,
    #                                                            usecols=(0,1,2,3,4,-4,-3,-2,-1))
    N, R11, R12, R21, R22, Rp11, Rp12, Rp21, Rp22 = S.T[[0,1,2,3,4,-4,-3,-2,-1]]
    # N to frequency in GHz: N*pi/W*c/(2*pi)/10**-9
    N = N*3.

    ax.semilogy(N, R11, "-", c=colors[0], lw=2.5, mec=colors[0], label=r'$R_{11}$')
    ax.semilogy(N, R22, "-", c=colors[1], lw=2.5, mec=colors[1], label=r'$R_{22}$')
    ax.semilogy(N, R21, "-", c=colors[2], lw=2.5, mec=colors[2], label=r'$R_{12}$')
    ax.semilogy(N, R12, "-", c=colors[4], lw=2.5, mec=colors[4], label=r'$R_{21}$')
    ax.semilogy(N, Rp11, "--", c=colors[0], lw=2.5, mec=colors[0], label=r'$R^\prime_{11}$')
    ax.semilogy(N, Rp22, "--", c=colors[1], lw=2.5, mec=colors[1], label=r'$R^\prime_{22}$')
    ax.semilogy(N, Rp21, "--", c=colors[2], lw=2.5, mec=colors[2], label=r'$R^\prime_{12}$')
    ax.semilogy(N, Rp12, "--", c=colors[4], lw=2.5, mec=colors[4], label=r'$R^\prime_{21}$')
    ax.set_xlabel(r"$\nu$ in GHz")
    ax.set_ylabel(r"$R_{nm}$, $R^\prime_{nm}$", labelpad=10)

    idx = np.where(np.isclose(N, 2.6*3, rtol=1e-2))
    print "T21/R"
    for r in R11, R21, R12, R22, Rp11, Rp12, Rp21, Rp22:
        print (T12[idx]/r[idx]).mean()


def main(swap=False):

    f, (ax1, ax2) = plt.subplots(figsize=(3,6), nrows=2)
    plt.subplots_adjust(hspace=0.55)

    S = np.loadtxt("S_matrix.dat")
    if swap:
        i = S[:, 6] < S[:, 7]
        for t in (1, 5, 9):
            S[i, t+1], S[i, t+2] = S[i, t+2], S[i, t+1]
            S[i, t  ], S[i, t+3] = S[i, t+3], S[i, t  ]

    N, T12 = plot_transmission(S, ax=ax1, swap=swap)
    plot_reflection(S, T12=T12, ax=ax2, swap=swap)

    for n, ax in enumerate((ax1, ax2)):
        legend_kwargs = dict(loc='upper center', ncol=4, frameon=False,
                             framealpha=0., labelspacing=-0.0)
        if n:
            ax_kwargs = dict(bbox_to_anchor=(0.5, 1.375))
        else:
            ax_kwargs = dict(bbox_to_anchor=(0.5, 1.25))

        legend_kwargs.update(ax_kwargs)
        legend = ax.legend(**legend_kwargs)

        ax.set_xticks(list(plt.xticks()[0]) + [7.8])
        ax.set_xticks([7.8], minor=True)
        ax.xaxis.grid(True, which='minor')

        # ax.set_xlim(N.min(), N.max())
        ax.set_xlim(6.9, 8.7)
        # ax.set_ylim(1e-8, 1)
        ax.set_ylim(1e-6, 1)

    # plt.tight_layout()
    # plt.show()
    plt.savefig("S_vs_N.pdf", bbox_extra_artist=legend, bbox_inches='tight')


if __name__ == '__main__':
    argh.dispatch_command(main)
