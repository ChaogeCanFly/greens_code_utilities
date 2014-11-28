#!/usr/bin/env python2.7

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import subprocess
import sys

import argh

import ep.profile
from ep.waveguide import Waveguide
import bloch
import helpers


def smooth_eigensystem(K_0, K_1, Chi_0, Chi_1, eps=2e-2, plot=True,
                       verbose=True):
    """Find discontinuities in the eigenvalues and reorder the eigensystem such
    that a smooth spectrum is obtained.

    Based on the (absolute) differences between the array components of K_0 and
    K_1, the maximum jumping value is determined (corresponding to a
    discontinuity from the modulo operation that is to be ignored in the
    following procedure). The arrays are then reassembled at points where
    jumps larger than eps have been found. We note that in a conservative
    system, one does not have to worry about continuous phases of the
    eigenfunctions, and, if gain/loss is introduced, the eigensystem is already
    sorted.

        Parameters:
        -----------
            K_0, K_1: list or (N,) ndarray
                eigenvalue list
            Chi_0, Chi_1: list or (N,...) ndarray
                eigenvector list
            eps: float
                maximum jump to be tolerated in the eigenvalues
            plot: bool
                whether to plot the spectrum before and after the smoothing
            verbose: bool
                whether to print additional output

        Returns:
        --------
            K_0, K_1: (N,) ndarray
            Chi_0, Chi_1: (N,...) ndarray
    """

    K_0, K_1 = [ np.array(z) for z in K_0, K_1]

    if plot:
        f, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.plot(K_0.real, "r-")
        ax1.plot(K_0.imag, "g-")
        ax1.plot(K_1.real, "r--")
        ax1.plot(K_1.imag, "g--")

    K_0_diff, K_1_diff = [ abs(np.diff(k)) for k in K_0, K_1 ]
    if K_0_diff.max() > K_1_diff.max():
        diff_max = K_0_diff.max()
        diff = K_0_diff
    else:
        diff_max = K_1_diff.max()
        diff = K_1_diff

    epsilon = 2.e-2
    jump = np.logical_and(diff > epsilon, diff != diff_max)

    if verbose:
        print "diff(K_n), jump-mask"
        for d, j in zip(diff, jump):
            print d, j

    for n in np.where(jump)[0]:
        K_0, K_1 = (np.concatenate((K_0[:n+1], K_1[n+1:])),
                    np.concatenate((K_1[:n+1], K_0[n+1:])))
        Chi_0, Chi_1 = (np.concatenate((Chi_0[:n+1], Chi_1[n+1:])),
                        np.concatenate((Chi_1[:n+1], Chi_0[n+1:])))

    Chi_0, Chi_1 = [ np.array(z) for z in Chi_0, Chi_1]

    if plot:
        ax2.plot(K_0.real, "r-")
        ax2.plot(K_0.imag, "g-")
        ax2.plot(K_1.real, "r--")
        ax2.plot(K_1.imag, "g--")
        plt.show()

    return K_0, K_1, Chi_0, Chi_1


def get_loop_eigenfunction(N=1.05, eta=0.0, L=5., init_phase=-0.05, eps=0.05,
                           nx=100, loop_direction="+", loop_type='Bell',
                           mpi=False, pphw=100):
    """Return the instantaneous eigenfunctions and eigenvectors for each step
    in a parameter space loop."""

    # get input.xml
    greens_path = os.environ.get('GREENS_CODE_XML')
    XML = os.path.join(greens_path, "input_periodic_cell.xml")


    wg_kwargs = {'N': N,
                 'eta': eta,
                 'L': L,
                 'init_phase': init_phase,
                 'loop_direction': loop_direction,
                 'loop_type': loop_type}
    WG = Waveguide(**wg_kwargs)
    WG.x_EP = eps
    WG.solve_ODE()

    x = np.linspace(0, L, nx)
    y = np.linspace(0, 1., (pphw*N+1))
    eps, delta = WG.get_cycle_parameters(x)

    K_0, K_1, Chi_0, Chi_1 = [ list() for n in range(4) ]

    for n, (xn, epsn, deltan) in enumerate(zip(x, eps, delta)):

        ID = "n_{:03}_xn_{:08.4f}".format(n, xn)
        print ID

        # prepare waveguide and profile
        profile_kwargs = {'eps': epsn,
                          'delta': deltan,
                          'pphw': pphw,
                          'input_xml': XML,
                          'custom_directory': os.getcwd(),
                          'neumann': 1}
        wg_kwargs_n = {'N': N,
                       'eta': eta,
                       'L': 2*np.pi/(WG.kr + deltan),
                       'init_phase': 0.0*init_phase,
                       'loop_direction': loop_direction,
                       'loop_type': 'Constant'}

        profile_kwargs.update(wg_kwargs_n)
        ep.profile.Generate_Profiles(**profile_kwargs)

        # run code
        if mpi:
            cmd = "mpirun -np 4 solve_xml_mumps dev"
        else:
            cmd = "solve_xml_mumps dev"
        greens_code = subprocess.Popen(cmd.split(),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
        greens_code.communicate()

        for file in glob.glob("N_*profile"):
            if "lower" in file:
                shutil.move(file, ID + ".lower_profile")
            if "upper" in file:
                shutil.move(file, ID + ".upper_profile")

        shutil.move("upper.dat", ID + ".upper_dat")
        shutil.move("lower.dat", ID + ".lower_dat")

        # get Bloch eigensystem
        K, _, ev, _, v, _ = bloch.get_eigensystem(return_eigenvectors=True,
                                                  return_velocities=True,
                                                  verbose=False, fold_back=False)
        if v[0] < 0. or v[1] < 0.:
            sys.exit("Error: group velocities are negative!")

        K0, K1 = K[0], K[1]
        ev0, ev1 = ev[0, :], ev[1, :]

        K_0.append(K0)
        K_1.append(K1)
        Chi_0.append(ev0)
        Chi_1.append(ev1)

        z = ev0.view(dtype=float)
        np.savetxt(ID + ".dat", zip(y, ev0.real, ev0.imag,
                                    np.ones_like(z)*K0.real,
                                    np.ones_like(z)*K0.imag,
                                    ev1.real, ev1.imag,
                                    np.ones_like(z)*K1.real,
                                    np.ones_like(z)*K1.imag),
                   header=('y Re(ev0) Im(ev0) Re(K0) Im(K0) Re(ev1)'
                           'Im(ev1) Re(K1) Im(K1)'))

        shutil.copyfile("pic.geometry.sine_boundary.1.jpg", ID + ".jpg")

        print "xn", xn, "epsn", epsn, "deltan", deltan, "K0", K0, "K1", K1

    K_0, K_1, Chi_0, Chi_1 = smooth_eigensystem(K_0, K_1, Chi_0, Chi_1,
                                                eps=2e-2, plot=False)
    Chi_0, Chi_1 = [ np.array(z).T for z in Chi_0, Chi_1]

    part = np.real

    X, Y = np.meshgrid(x, y)

    plt.clf()
    Z = part(Chi_0 * np.exp(1j*K_0*x))
    p = plt.pcolormesh(X, Y, Z)
    plt.colorbar(p)
    plt.savefig("Chi_0.png")

    plt.clf()
    Z = part(Chi_1 * np.exp(1j*K_1*x))
    p = plt.pcolormesh(X, Y, Z)
    plt.colorbar(p)
    plt.savefig("Chi_1.png")

    # effective model predictons
    Chi_0_eff, Chi_1_eff = WG.eVecs_r[:,:,0], WG.eVecs_r[:,:,1]
    K_0_eff, K_1_eff = WG.eVals[:,0], WG.eVals[:,1]

    # ------------------------------------------------------------------------
    # if 0:
    #     plt.clf()
    #     incr = 100
    #     plt.plot(WG.t[::incr], abs(Chi_0_eff[:,0][::incr]), "r--")
    #     plt.plot(WG.t[::incr], abs(Chi_0_eff[:,1][::incr]), "g-")
    #     plt.plot(WG.t[::incr], abs(Chi_1_eff[:,0][::incr]), "y--")
    #     plt.plot(WG.t[::incr], abs(Chi_1_eff[:,1][::incr]), "k-")
    #     plt.show()
    # ------------------------------------------------------------------------

    X_eff, Y_eff = np.meshgrid(WG.t, y)

    Chi_0_eff_0 = np.outer(Chi_0_eff[:,0], 1*np.ones_like(y))
    Chi_0_eff_1 = np.outer(Chi_0_eff[:,1], #*np.exp(-1j*WG.kr*WG.t),
                           np.sqrt(2.*WG.k0/WG.k1)*np.cos(np.pi*y))
    Chi_0_eff = Chi_0_eff_0 + Chi_0_eff_1

    Chi_1_eff_0 = np.outer(Chi_1_eff[:,0], 1*np.ones_like(y))
    Chi_1_eff_1 = np.outer(Chi_1_eff[:,1], #*np.exp(-1j*WG.kr*WG.t),
                           np.sqrt(2.*WG.k0/WG.k1)*np.cos(np.pi*y))
    Chi_1_eff = Chi_1_eff_0 + Chi_1_eff_1

    plt.clf()
    Z_eff = part(Chi_0_eff.T * np.exp(1j*(K_0_eff*WG.t)))
    p = plt.pcolormesh(X_eff, Y_eff, Z_eff)
    plt.colorbar(p)
    plt.savefig("Chi_0_eff.png")

    plt.clf()
    Z_eff = part(Chi_1_eff.T * np.exp(1j*(K_1_eff*WG.t)))
    p = plt.pcolormesh(X_eff, Y_eff, Z_eff)
    plt.colorbar(p)
    plt.savefig("Chi_1_eff.png")


if __name__ == '__main__':
    argh.dispatch_command(get_loop_eigenfunction)