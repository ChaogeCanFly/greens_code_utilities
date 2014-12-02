#!/usr/bin/env python2.7

import glob
import matplotlib.pyplot as plt
import multiprocessing
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


def smooth_eigensystem(K_0, K_1, Chi_0, Chi_1, eps=2e-3, plot=True):
    """Find discontinuities in the eigenvalues and reorder the eigensystem to
    obtain a smooth spectrum.

    Based on the (absolute) differences between the array components of K_0 and
    K_1, the maximum jumping value is determined (corresponding to a
    discontinuity from the modulo operation that has to be ignored in the
    following procedure). The arrays are then reassembled at points where
    jumps larger than eps have been found. We note that in a conservative
    system, one does not have to worry about continuous phases of the
    eigenfunctions, and, if gain/loss are introduced, the eigensystem is
    already sorted.

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

    # jump = np.logical_and(diff > eps, diff != diff_max)
    jump = diff > eps

    for n in np.where(jump)[0]:
        K_0, K_1 = (np.concatenate((K_0[:n+1], K_1[n+1:])),
                    np.concatenate((K_1[:n+1], K_0[n+1:])))
        Chi_0, Chi_1 = (np.concatenate((Chi_0[:n+1], Chi_1[n+1:])),
                        np.concatenate((Chi_1[:n+1], Chi_0[n+1:])))

    if plot:
        ax2.plot(K_0.real, "r-")
        ax2.plot(K_0.imag, "g-")
        ax2.plot(K_1.real, "r--")
        ax2.plot(K_1.imag, "g--")
        plt.show()

    return K_0, K_1, Chi_0, Chi_1


# def run_single_job(n, xn, epsn, deltan, eta=0.0, pphw=100, XML='input.xml',
# def run_single_job(args, eta=0.0, pphw=100, XML='input.xml',
                   # N=1.05, WG=Waveguide(), loop_direction='-'):
def run_single_job(args):
    """."""
    n, xn, epsn, deltan, eta, pphw, XML, N, WG, loop_direction = args
    # n, xn, epsn, deltan = args
    ID = "n_{:03}_xn_{:08.4f}".format(n, xn)
    print
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
                   'init_phase': 0.0,
                   'loop_direction': loop_direction,
                   'loop_type': 'Constant'}

    profile_kwargs.update(wg_kwargs_n)
    ep.profile.Generate_Profiles(**profile_kwargs)

    # run code
    cmd = "solve_xml_mumps dev"
    greens_code = subprocess.Popen(cmd.split(),
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
    greens_code.communicate()

    # for file in glob.glob("N_*profile"):
    #     if "lower" in file:
    #         shutil.move(file, ID + ".lower_profile")
    #     if "upper" in file:
    #         shutil.move(file, ID + ".upper_profile")
    #
    # shutil.move("upper.dat", ID + ".upper_dat")
    # shutil.move("lower.dat", ID + ".lower_dat")

    # get Bloch eigensystem
    K, _, ev, _, v, _ = bloch.get_eigensystem(return_eigenvectors=True,
                                              return_velocities=True,
                                              verbose=True,
                                              fold_back=True)
    if np.real(v[0]) < 0. or np.real(v[1]) < 0.:
        sys.exit("Error: group velocities are negative!")

    K0, K1 = K[0], K[1]
    ev0, ev1 = ev[0,:], ev[1,:]

    z = ev0.view(dtype=float)
    np.savetxt(ID + ".dat", zip(ev0.real, ev0.imag,
                                np.ones_like(z)*K0.real,
                                np.ones_like(z)*K0.imag,
                                ev1.real, ev1.imag,
                                np.ones_like(z)*K1.real,
                                np.ones_like(z)*K1.imag),
               header=('y Re(ev0) Im(ev0) Re(K0) Im(K0) Re(ev1)'
                       'Im(ev1) Re(K1) Im(K1)'))

    shutil.copyfile("pic.geometry.sine_boundary.1.jpg", ID + ".jpg")

    print "xn", xn, "epsn", epsn, "deltan", deltan, "K0", K0, "K1", K1
    return K0, K1, ev0, ev1


def get_loop_eigenfunction(N=1.05, eta=0.0, L=5., init_phase=0.0, eps=0.05,
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

    pool = multiprocessing.Pool(processes=4)
    # job_args = enumerate(zip(x, eps, delta))
    job_kwargs = {'eta': eta,
                  'pphw': pphw,
                  'XML': XML,
                  'N': N,
                  'WG': WG,
                  'loop_direction': loop_direction}
    # for n, (xn, epsn, deltan) in enumerate(zip(x, eps, delta)):
    #     print n
        # K0, K1, ev0, ev1 = pool.apply_async(run_single_job, args=(n, xn, epsn, deltan),
        #                               kwds=job_kwargs).get()
        # K_0.append(K0)
        # K_1.append(K1)
        # Chi_0.append(ev0)
        # Chi_1.append(ev1)

    job_list = [ (n, xn, epsn, deltan, eta, pphw, XML, N, WG, loop_direction) for n, (xn, epsn, deltan) in enumerate(zip(x, eps, delta)) ]
    print job_list

    pool.map(run_single_job, job_list)
    K_0, K_1, Chi_0, Chi_1 = results
    # K_0.append(K0)
    # K_1.append(K1)
    # Chi_0.append(ev0)
    # Chi_1.append(ev1)

    K_0, K_1, Chi_0, Chi_1 = smooth_eigensystem(K_0, K_1, Chi_0, Chi_1,
                                                eps=2e-2, plot=False)
    Chi_0, Chi_1 = [ np.array(c).T for c in Chi_0, Chi_1]

    # test: unwrapping --------------------------------------------------------
    L_range = 2*np.pi/(WG.kr + delta)  # make small error since L != r_nx*dx
    K_0 = np.unwrap(K_0.real*L_range)/L_range + 1j*K_0.imag
    K_1 = np.unwrap(K_1.real*L_range)/L_range + 1j*K_1.imag
    # -------------------------------------------------------------------------

    # effective model predictons
    K_0_eff, K_1_eff = WG.eVals[:,0], WG.eVals[:,1]

    # ------------------------------------------------------------------------
    # eigenvalues
    part = np.real
    if 1:
        plt.clf()
        f, (ax1, ax2, ax3) = plt.subplots(nrows=3)
        ax1.plot(x, part(K_0), "r-")
        ax1.plot(x, part(K_1), "g--")
        ax2.plot(WG.t, part(K_0_eff), "r-")
        ax2.plot(WG.t, part(K_1_eff), "g--")
        ax3.plot(x, abs(K_1 - K_0), "k-")
        ax3.plot(WG.t, abs(K_1_eff.real - K_0_eff.real), "k--")
        plt.savefig("eigenvalues.png")
    # ------------------------------------------------------------------------


if __name__ == '__main__':
    argh.dispatch_command(get_loop_eigenfunction)
