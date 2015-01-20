#!/usr/bin/env python2.7

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from scipy.interpolate import interp1d
import shutil
import subprocess
import sys
import traceback

import argh

import ep.profile
from ep.waveguide import Waveguide
import bloch


def smooth_eigensystem(K_0, K_1, Chi_0, Chi_1, eps=2e-2, plot=True):
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
                maximum jump-height of signal
            plot: bool
                whether to plot the spectrum before and after the smoothing

        Returns:
        --------
            K_0, K_1: (N,) ndarray
            Chi_0, Chi_1: (N,...) ndarray
    """

    K_0, K_1, Chi_0, Chi_1 = [ np.asarray(z) for z in K_0, K_1, Chi_0, Chi_1 ]

    if plot:
        f, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.plot(K_0.real, "r-")
        ax1.plot(K_0.imag, "g-")
        ax1.plot(K_1.real, "r--")
        ax1.plot(K_1.imag, "g--")

    # -------------------------------------------------------------------------
    # the following is not necessary if a small eta is chosen, s.t.
    # the eigenvalues can be ordered w.r.t. the magnitude of the imaginary part
    #
    # sort all eigenvalues s.t. one eigenvalue is always larger than the other
    # jump = K_0.real < K_1.real
    # K_0[jump], K_1[jump] = K_1[jump], K_0[jump]
    # Chi_0[jump], Chi_1[jump] = Chi_1[jump], Chi_0[jump]
    #
    # search for points where diff(K_n) is larger than a threshold epsilon
    # if, at these points, |K_0[n] - K_1[n+1]| < |diff(K_n)|, switch
    #
    # for n in np.where(real_diff > 5e-2)[0]:
    #     real_diff = np.abs(np.diff(np.real(K_1)))
    #     cross_diff = np.abs(np.real(K_0)[n] - np.real(K_1)[n+1])
    #     if cross_diff < real_diff[n]:
    #         K_0, K_1 = (np.concatenate((K_0[:n+1], K_1[n+1:])),
    #                     np.concatenate((K_1[:n+1], K_0[n+1:])))
    #         Chi_0, Chi_1 = (np.concatenate((Chi_0[:n+1,:], Chi_1[n+1:,:])),
    #                         np.concatenate((Chi_1[:n+1,:], Chi_0[n+1:,:])))
    #
    # find minimum distance between eigenvalues and switch (only for eta = 0)
    # if abs(K_0 - K_1).min() < eps:
    # if not eps:
    #     n = np.argmin(abs(K_0 - K_1))
    #     K_0, K_1 = (np.concatenate((K_0[:n+1], K_1[n+1:])),
    #                 np.concatenate((K_1[:n+1], K_0[n+1:])))
    #     Chi_0, Chi_1 = (np.concatenate((Chi_0[:n+1], Chi_1[n+1:])),
    #                     np.concatenate((Chi_1[:n+1], Chi_0[n+1:])))
    # -------------------------------------------------------------------------

    abs_diff = np.abs(np.diff(np.abs(K_0)))
    real_diff = np.abs(np.diff(np.real(K_0)))

    for n in np.where(real_diff > eps)[0]:
        cross_diff_0 = np.abs(np.real(K_0)[n] - np.real(K_1)[n+1])
        cross_diff_1 = np.abs(np.real(K_1)[n] - np.real(K_0)[n+1])
        if abs(cross_diff_0 - cross_diff_1) < eps:
            K_0, K_1 = (np.concatenate((K_0[:n+1], K_1[n+1:])),
                        np.concatenate((K_1[:n+1], K_0[n+1:])))
            Chi_0, Chi_1 = (np.concatenate((Chi_0[:n+1,:], Chi_1[n+1:,:])),
                            np.concatenate((Chi_1[:n+1,:], Chi_0[n+1:,:])))

    # find gain-state:
    Gamma0, Gamma1  = [ np.sum(k.imag) for k in K_0, K_1 ]
    if Gamma0 < Gamma1:
        K_0, K_1, Chi_0, Chi_1 = K_1, K_0, Chi_1, Chi_0

    if plot:
        ax2.plot(K_0.real, "r-")
        ax2.plot(K_0.imag, "g-")
        ax2.plot(K_1.real, "r--")
        ax2.plot(K_1.imag, "g--")
        plt.show()

    return K_0, K_1, Chi_0, Chi_1


def run_single_job(n, xn, epsn, deltan, eta=None, pphw=None, XML=None, N=None,
                   WG=None, loop_direction=None, neumann=None):
    """Calculate the Bloch eigensystem in a separate directory and extract the
    eigenvalues and eigenvectors.

        Returns:
        --------
            K0, K1: complex
               eigenvalues
            ev0, ev1: (N,) ndarray
               eigenvectors
    """

    ID = "n_{:03}_xn_{:08.4f}".format(n, xn)
    print
    print ID

    CWD = os.getcwd()
    DIR = os.path.join(CWD, ID)
    os.makedirs(ID)
    os.chdir(DIR)

    # prepare waveguide and profile
    profile_kwargs = {'eps': epsn,
                      'delta': deltan,
                      'pphw': pphw,
                      'input_xml': XML,
                      'custom_directory': os.getcwd(),
                      'neumann': neumann}
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

    try:
        # get Bloch eigensystem
        K, _, ev, _, v, _ = bloch.get_eigensystem(return_eigenvectors=True,
                                                  return_velocities=True,
                                                  verbose=True,
                                                  neumann=neumann)
        # remove eigenvalue files
        # os.remove("Evecs.sine_boundary.dat")
        # os.remove("Evecs.sine_boundary.abs")

        # if np.real(v[0]) < 0. or np.real(v[1]) < 0.:
        #     sys.exit("Error: group velocities are negative!" + 180*"-")

        K0, K1 = K[0], K[1]
        ev0, ev1 = ev[0,:], ev[1,:]
        print "chi.shape", ev0.shape

        z = ev0.view(dtype=float)
        np.savetxt("eigensystem.dat", zip(ev0.real, ev0.imag,
                                    np.ones_like(z)*K0.real,
                                    np.ones_like(z)*K0.imag,
                                    ev1.real, ev1.imag,
                                    np.ones_like(z)*K1.real,
                                    np.ones_like(z)*K1.imag),
                    header=('y Re(ev0) Im(ev0) Re(K0) Im(K0) Re(ev1)'
                            'Im(ev1) Re(K1) Im(K1)'))
        os.chdir(CWD)
        # subprocess.call("gzip -r {}".format(DIR).split())
        shutil.rmtree(DIR)

        print "xn", xn, "epsn", epsn, "deltan", deltan, "K0", K0, "K1", K1
        return K0, K1, ev0, ev1

    except:
        os.chdir(CWD)
        print traceback.print_exc()
        return


def get_loop_eigenfunction(N=1.05, eta=0.0, L=5., d=1., eps=0.05, nx=None,
                           loop_direction="+", loop_type='Bell', init_state='a',
                           init_phase=0.0, mpi=False, pphw=100,
                           effective_model_only=False,
                           neumann=1):
    """Return the instantaneous eigenfunctions and eigenvectors for each step
    in a parameter space loop.

        Parameters:
        -----------
            N: float
                Number of open modes (floor(N)).
            eta: float
                Dissipation strength.
            L: float
                System length.
            d: float
                System width.
            eps: float
                Half-maximum boundary roughness.
            nx: int
                Number of slices to calculate. If None, determine nx via
                pphw and N automatically.
            loop_direction: str
                Loop direction of the parameter space trajectory. Allowed
                values: + or -.
            loop_type: str
                Trajectory shape in parameter space.
            init_state: str
                Initial state of the evolution in the effective 2x2 system.
            init_phase: float
                Initial phase in the trajectory.
            mpi: bool
                Whether to use the parallel greens_code version.
            pphw: int
                Points per half-wavelength.
            effective_model_only: bool
                Whether to only calculate the effective model predictions.
            neumann: int
                Whether to use Neumann or Dirichlet boundary conditions.
    """

    greens_path = os.environ.get('GREENS_CODE_XML')
    XML = os.path.join(greens_path, "input_periodic_cell.xml")

    wg_kwargs = {'N': N,
                 'eta': eta,
                 'L': L,
                 'init_phase': init_phase,
                 'init_state': init_state,
                 'loop_direction': loop_direction,
                 'loop_type': loop_type,
                 'neumann': neumann}
    WG = Waveguide(**wg_kwargs)
    WG.x_EP = eps
    _, b0, b1 = WG.solve_ODE()

    # prepare waveguide and profile -------------------------------------------
    profile_kwargs = {'eps': eps,
                      'pphw': pphw,
                      'input_xml': XML,
                      'custom_directory': os.getcwd(),
                      'neumann': neumann}
    profile_kwargs.update(wg_kwargs)

    ep.profile.Generate_Profiles(**profile_kwargs)

    for file in glob.glob("N_*profile"):
        if "lower" in file:
            shutil.move(file, "boundary.lower_profile")
        if "upper" in file:
            shutil.move(file, "boundary.upper_profile")
    # -------------------------------------------------------------------------

    # trajectories ------------------------------------------------------------
    if 0:
        f, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.semilogy(WG.t, abs(b0), "r-")
        ax1.semilogy(WG.t, abs(b1), "g-")

        wg_kwargs['loop_direction'] = '+'
        WG = Waveguide(**wg_kwargs)
        WG.x_EP = eps
        _, b0, b1 = WG.solve_ODE()

        ax1.semilogy(WG.t, abs(b0[::-1]), "r--")
        ax1.semilogy(WG.t, abs(b1[::-1]), "g--")

        ax2.plot(WG.t, WG.eVals[:,0].real, "r-")
        ax2.plot(WG.t, WG.eVals[:,1].real, "g-")
        ax2.plot(WG.t, WG.eVals[:,0].imag, "r--")
        ax2.plot(WG.t, WG.eVals[:,1].imag, "g--")
        # plt.savefig("evals_trajectories.png")
        plt.show()
    # -------------------------------------------------------------------------

    # change nx and ny accoring to pphw and modes -----------------------------
    nyout = pphw*N
    if nx is None:
        nx = int(L*(nyout+1.))
        print "nx:", nx
    ny = int(d*(nyout+1.))
    # -------------------------------------------------------------------------

    x = np.linspace(0, L, nx)
    y = np.linspace(0, d, ny)
    eps, delta = WG.get_cycle_parameters(x)

    K_0, K_1, Chi_0, Chi_1 = [ list() for n in range(4) ]

    job_kwargs = {'eta': eta,
                  'pphw': pphw,
                  'XML': XML,
                  'N': N,
                  'WG': WG,
                  'loop_direction': loop_direction,
                  'neumann': neumann}

    # serialized version:
    # results = []
    # for n, (xn, epsn, deltan) in enumerate(zip(x, eps, delta)):
    #     run_single_job(n, xn, epsn, deltan, **job_kwargs)

    # alternative parallelization:
    # job_list = [ (n, xn, epsn, deltan, eta, pphw, XML, N, WG, loop_direction)
    #               for n, (xn, epsn, deltan) in enumerate(zip(x, eps, delta)) ]
    # results = pool.map(run_single_job, job_list)

    if not effective_model_only:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = [ pool.apply_async(run_single_job, args=(n, xn, epsn, deltan),
                                    kwds=job_kwargs)
                    for n, (xn, epsn, deltan) in enumerate(zip(x, eps, delta)) ]
        results = [ p.get() for p in results ]

        # properly unpack results
        for res in results:
            if res is None:
                # reuse last set of eigenvalues/eigenvectors if something went
                # wrong in the function call of run_single_job
                print "Warning: calculation failed."
            else:
                K0, K1, ev0, ev1 = res
            K_0.append(K0)
            K_1.append(K1)
            Chi_0.append(ev0)
            Chi_1.append(ev1)

        # -------------------------------------------------------------------------
        # numerical data

        K_0, K_1, Chi_0, Chi_1 = [ np.asarray(z) for z in K_0, K_1, Chi_0, Chi_1 ]

        # smooth
        K_0, K_1, Chi_0, Chi_1 = smooth_eigensystem(K_0, K_1, Chi_0, Chi_1,
                                                    eps=WG.x_EP, plot=False)

        # transpose array!
        Chi_0, Chi_1 = [ np.array(c).T for c in Chi_0, Chi_1 ]

        # unwrapp phase
        G = delta + WG.kr
        L_range = 2*np.pi/G  # make small error since L != r_nx*dx
        K_0 = np.unwrap(K_0.real*L_range)/L_range + 1j*K_0.imag
        K_1 = np.unwrap(K_1.real*L_range)/L_range + 1j*K_1.imag

        # TODO: handle case for eps = 0.0
        # remove discontinuous second order derivative (if we cross the DP)
        # plt.plot(np.diff(np.abs(K_0), 2))
        # plt.show()

        # assemble eigenvectors
        Chi_0 *= np.exp(1j*K_0*x)
        Chi_1 *= np.exp(1j*K_1*x) * np.exp(-1j*WG.kr*x)
        # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # effective model predictons

    # get eigensystem
    Chi_0_eff, Chi_1_eff = WG.eVecs_r[:,:,0], WG.eVecs_r[:,:,1]
    # Chi_0_eff, Chi_1_eff = b0, b1
    K_0_eff, K_1_eff = WG.eVals[:,0], WG.eVals[:,1]

    # interpolate
    K_0_eff = (interp1d(WG.t, K_0_eff.real)(x) +
                1j*interp1d(WG.t, K_0_eff.imag)(x))
    K_1_eff = (interp1d(WG.t, K_1_eff.real)(x) +
                1j*interp1d(WG.t, K_1_eff.imag)(x))

    Chi_0_eff = [ (interp1d(WG.t, Chi_0_eff[:,n].real)(x) +
                    1j*interp1d(WG.t, Chi_0_eff[:,n].imag)(x)) for n in 0, 1 ]
    Chi_1_eff = [ (interp1d(WG.t, Chi_1_eff[:,n].real)(x) +
                    1j*interp1d(WG.t, Chi_1_eff[:,n].imag)(x)) for n in 0, 1 ]
    # Chi_0_eff = [ (interp1d(WG.t, Chi_0_eff[:].real)(x) +
    #                 1j*interp1d(WG.t, Chi_0_eff[:].imag)(x)) for n in 0, 1 ]
    # Chi_1_eff = [ (interp1d(WG.t, Chi_1_eff[:].real)(x) +
    #                 1j*interp1d(WG.t, Chi_1_eff[:].imag)(x)) for n in 0, 1 ]
    Chi_0_eff, Chi_1_eff = [ np.array(c).T for c in Chi_0_eff, Chi_1_eff ]


    # fold back
    G = delta + WG.kr
    L_range = 2*np.pi/G  # make small error since L != r_nx*dx

    # np.savetxt("output_data.dat", zip(G, L_range, K_0_eff.real, K_1_eff.real,
    #                                   K_0.real, K_1.real),
    #            header='G L_range K_0_eff K_1_eff K_0 K_1')
    K_0_eff = ((-K_0_eff.real + G/2.) % G - G/2.) + 1j*K_0_eff.imag
    K_1_eff = ((-K_1_eff.real + G/2.) % G - G/2.) + 1j*K_1_eff.imag

    # unwrapp phase
    K_0_eff = np.unwrap(K_0_eff.real*L_range)/L_range + 1j*K_0_eff.imag
    K_1_eff = np.unwrap(K_1_eff.real*L_range)/L_range + 1j*K_1_eff.imag

    # assemble effective model eigenvectors
    Chi_0_eff[:,0] *= np.exp(1j*K_0_eff*x)
    Chi_0_eff[:,1] *= np.exp(1j*K_0_eff*x)
    Chi_1_eff[:,0] *= np.exp(1j*K_1_eff*x)
    Chi_1_eff[:,1] *= np.exp(1j*K_1_eff*x)

    # no additional factor of np.exp(-i*kr*x) since we unwrap the phase,
    # corresponding to K_n -> K_n + n*G
    if neumann:
        Chi_0_eff_0 = np.outer(Chi_0_eff[:,0], 1.*np.ones_like(y))
        Chi_0_eff_1 = np.outer(Chi_0_eff[:,1], #*np.exp(-1j*WG.kr*x),
                            np.sqrt(2.*WG.k0/WG.k1)*np.cos(np.pi*y))
        Chi_0_eff = Chi_0_eff_0 + Chi_0_eff_1

        Chi_1_eff_0 = np.outer(Chi_1_eff[:,0], 1.*np.ones_like(y))
        Chi_1_eff_1 = np.outer(Chi_1_eff[:,1], #*np.exp(-1j*WG.kr*x),
                            np.sqrt(2.*WG.k0/WG.k1)*np.cos(np.pi*y))
        Chi_1_eff = Chi_1_eff_0 + Chi_1_eff_1
    else:
        Chi_0_eff_0 = np.outer(Chi_0_eff[:,0], np.sin(np.pi*y))
        Chi_0_eff_1 = np.outer(Chi_0_eff[:,1], #*np.exp(-1j*WGn.kr*xnn),
                            np.sqrt(WG.k1/WG.k0)*np.sin(2*np.pi*y))
        Chi_0_eff = Chi_0_eff_0 + Chi_0_eff_1

        Chi_1_eff_0 = np.outer(Chi_1_eff[:,0], np.sin(np.pi*y))
        Chi_1_eff_1 = np.outer(Chi_1_eff[:,1], #*np.exp(-1j*WGn.kr*xnn),
                            np.sqrt(WG.k1/WG.k0)*np.sin(2*np.pi*y))
        Chi_1_eff = Chi_1_eff_0 + Chi_1_eff_1

    Chi_0_eff, Chi_1_eff = [ c.T for c in Chi_0_eff, Chi_1_eff ]
    # -------------------------------------------------------------------------

    # eigenvalues -------------------------------------------------------------
    if 1:
        plt.clf()
        f, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(8,6), dpi=80)
        prop = {'size': 12}

        if not effective_model_only:
            ax1.set_title(r"Numerical eigenvalues $K_n$")
            ax1.plot(x, K_0.real, "r-", label=r"$\Re{K_0}$")
            ax1.plot(x, K_0.imag, "b-", label=r"$\Im{K_0}$")
            ax1.plot(x, K_1.real, "r--", label=r"$\Re{K_1}$")
            ax1.plot(x, K_1.imag, "b--", label=r"$\Im{K_1}$")
            ax1.set_xlabel(r"$x$")
            l1 = ax1.legend(bbox_to_anchor=(1.3,1.075), prop=prop)

        ax2.set_title(r"Effective model eigenvalues $K^{\mathrm{eff}}_n$")
        ax2.plot(x, K_0_eff.real, "r-", label=r"$\Re{K^{\mathrm{eff}}_0}$")
        ax2.plot(x, K_0_eff.imag, "b-", label=r"$\Im{K^{\mathrm{eff}}_0}$")
        ax2.plot(x, K_1_eff.real, "r--", label=r"$\Re{K^{\mathrm{eff}}_1}$")
        ax2.plot(x, K_1_eff.imag, "b--", label=r"$\Im{K^{\mathrm{eff}}_1}$")
        ax2.set_xlabel(r"$x$")
        l2 = ax2.legend(bbox_to_anchor=(1.3,1.075), prop=prop)
        extra_artist=[l2]

        if not effective_model_only:
            ax3.set_title("Comparison")
            ax3.plot(x, abs(K_0 - K_0_eff)**2, "k-", label=r"$|K_0 - K^{\mathrm{eff}}_0|^2$")
            ax3.plot(x, abs(K_1 - K_1_eff)**2, "k--", label=r"$|K_1 - K^{\mathrm{eff}}_1|^2$")
            ax3.set_xlabel(r"$x$")
            l3 = ax3.legend(bbox_to_anchor=(1.3,1.075), prop=prop)
            extra_artist=[l3]

        plt.tight_layout()
        plt.savefig("eigenvalues.png", bbox_extra_artists=(extra_artist), 
                    bbox_inches='tight')
    # ------------------------------------------------------------------------

    # save potential ----------------------------------------------------------
    # for n, c in enumerate((Chi_0, Chi_1)):
    #     c = np.abs(c).flatten(order='F')
    #     nfile = range(len(c))
    #     np.savetxt("potential_imag_{}.dat".format(n),
    #               zip(nfile, c), fmt='%i %10.6f')
    #     c = (c.max() - c)/c.max()
    #     np.savetxt("potential_imag_{}_normalized.dat".format(n),
    #                zip(nfile, c), fmt='%i %10.6f')

    wavefunctions = [Chi_0_eff, Chi_1_eff]
    names = ["Chi_0_eff", "Chi_1_eff"]
    if not effective_model_only:
        wavefunctions += [Chi_0, Chi_1]
        names += ["Chi_0", "Chi_1"]

    for n, c in zip(names, wavefunctions):
        np.savetxt("potential_{}.dat".format(n), c)

        c = np.abs(c).flatten(order='F')
        nfile = range(len(c))
        np.savetxt("potential_{}_imag.dat".format(n),
                   zip(nfile, c), fmt='%i %10.6f')
        c = (c.max() - c)/c.max()
        np.savetxt("potential_{}_imag_normalized.dat".format(n),
                   zip(nfile, c), fmt='%i %10.6f')
    # -------------------------------------------------------------------------

    X, Y = np.meshgrid(x, y)
    for part in np.abs, np.angle, np.real, np.imag:
        if part == np.abs:
            cmap = 'Reds'
        else:
            cmap = 'RdBu_r'

        if not effective_model_only:
            plt.clf()
            Z = part(Chi_0)
            p = plt.pcolormesh(X, Y, Z, cmap=cmap)
            plt.colorbar(p)
            plt.savefig("Chi_0_{0.__name__}.png".format(part))

            plt.clf()
            Z = part(Chi_1)
            p = plt.pcolormesh(X, Y, Z, cmap=cmap)
            plt.colorbar(p)
            plt.savefig("Chi_1_{0.__name__}.png".format(part))

        plt.clf()
        Z_eff = part(Chi_0_eff)
        p = plt.pcolormesh(X, Y[::-1,:], Z_eff, cmap=cmap)
        plt.colorbar(p)
        plt.savefig("Chi_0_eff_{0.__name__}.png".format(part))

        plt.clf()
        Z_eff = part(-1j*Chi_1_eff)
        p = plt.pcolormesh(X, Y[::-1,:], Z_eff, cmap=cmap)
        plt.colorbar(p)
        plt.savefig("Chi_1_eff_{0.__name__}.png".format(part))


if __name__ == '__main__':
    argh.dispatch_command(get_loop_eigenfunction)
