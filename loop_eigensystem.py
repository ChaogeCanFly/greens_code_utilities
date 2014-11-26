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


def get_loop_eigenfunction(N=1.05, eta=0.0, L=5., init_phase=-0.05, eps=0.025,
                           nx=100,
                           loop_direction="+", loop_type='Bell', pphw=100):
    """Return the instantaneous eigenfunctions and eigenvectors for each step
    in a parameter space loop."""

    # get input.xml
    greens_path = os.environ.get('GREENS_CODE_XML')
    XML = os.path.join(greens_path, "input_periodic_cell.xml")

    x = np.linspace(0, L, nx)
    y = np.linspace(0, 1., (pphw*N+1))

    wg_kwargs = {'N': N,
                 'eta': eta,
                 'L': L,
                 'init_phase': init_phase,
                 'loop_direction': loop_direction,
                 'loop_type': loop_type}
    WG = Waveguide(**wg_kwargs)
    WG.x_EP = eps
    WG.solve_ODE()

    eps, delta = WG.get_cycle_parameters(x)

    Bloch_data = []
    K_0, K_1, Chi_0, Chi_1 = [ [] for n in range(4) ]
    for n, (xn, epsn, deltan) in enumerate(zip(x, eps, delta)):

        ID = "n_{}_xn_{:08.4f}".format(n, xn)
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
        # cmd = "mpirun -np 4 solve_xml_mumps dev"
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
                                                  verbose=False)
        if v[0] < 0. or v[1] < 0.:
            sys.exit("Error: group velocities are negative!")

        K0, K1 = K[0], K[1]
        ev0, ev1 = ev[0, :], ev[1, :]

        diff_0 = np.abs(np.diff(ev0)).sum()
        diff_1 = np.abs(np.diff(ev1)).sum()
        print 'diff before', diff_0
        print 'diff before', diff_1
        # if K0 > K1 and xn < L/2.:
        #     K0, K1 = K1, K0
        if diff_0 > diff_1 and xn < L/2.:
            ev0, ev1 = ev1, ev0

        # if K0 < K1 and xn > L/2.:
        #     K0, K1 = K1, K0
        if diff_0 < diff_1 and xn > L/2.:
            ev0, ev1 = ev1, ev0

        diff_0 = np.abs(np.diff(ev0)).sum()
        diff_1 = np.abs(np.diff(ev1)).sum()
        print 'diff after', diff_0
        print 'diff after', diff_1

        Bloch_data.append([xn, epsn, deltan,
                           K0, K1, ev0, ev1])
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

    # full system
    Chi_0, Chi_1, K_0, K_1 = [ np.array(z).T for z in Chi_0, Chi_1, K_0, K_1]
    # Chi_0 = np.array(Chi_0).T
    # Chi_1 = np.array(Chi_1).T

    X, Y = np.meshgrid(x, y)
    plt.clf()
    plt.pcolormesh(X, Y, np.real(Chi_0)) # * np.exp(1j*K_0*x)))
    plt.savefig("Chi_0.png")
    plt.clf()
    plt.pcolormesh(X, Y, np.real(Chi_1)) # * np.exp(1j*K_1*x)))
    plt.savefig("Chi_1.png")

    # effective model predictons
    Chi_0_eff, Chi_1_eff = WG.eVecs_r[:,:,0], WG.eVecs_r[:,:,1]
    K_0_eff, K_1_eff = WG.eVals[:,0], WG.eVals[:,1]

    Chi_0_eff_0 = np.outer(Chi_0_eff[:,0], np.ones_like(y))
    Chi_0_eff_1 = np.outer(Chi_0_eff[:,1]*np.exp(-1j*WG.kr*WG.t),
                           np.sqrt(2.*WG.k0/WG.k1)*np.cos(np.pi*y))
    Chi_0_eff = Chi_0_eff_0 + Chi_0_eff_1

    Chi_1_eff_0 = np.outer(Chi_1_eff[:,0], np.ones_like(y))
    Chi_1_eff_1 = np.outer(Chi_1_eff[:,1]*np.exp(-1j*WG.kr*WG.t),
                           np.sqrt(2.*WG.k0/WG.k1)*np.cos(np.pi*y))
    Chi_1_eff = Chi_1_eff_0 + Chi_1_eff_1

    X_eff, Y_eff = np.meshgrid(WG.t, y)
    plt.clf()
    plt.pcolormesh(X_eff, Y_eff, np.real(Chi_0_eff.T)) # * np.exp(1j*(K_0_eff*WG.t))))
    # plt.pcolormesh(X_eff, Y_eff, np.real(Chi_0_eff.T * np.exp(1j*(K_0_eff*X_eff))))
    plt.savefig("Chi_0_eff.png")
    plt.clf()
    plt.pcolormesh(X_eff, Y_eff, np.real(Chi_1_eff.T)) # * np.exp(1j*K_1_eff*WG.t)))
    # plt.pcolormesh(X_eff, Y_eff, np.real(Chi_1_eff.T * np.exp(1j*K_1_eff*X_eff)))
    plt.savefig("Chi_1_eff.png")


if __name__ == '__main__':
    argh.dispatch_command(get_loop_eigenfunction)
