#!/usr/bin/env python2.7

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import subprocess

import argh

import ep.profile
from ep.waveguide import Waveguide
import bloch
import helpers


def get_loop_eigenfunction(N=1.05, eta=0.0, L=5., init_phase=-0.05, eps=0.05,
                           loop_direction="+", loop_type='Bell', pphw=100):

    # get input.xml
    greens_path = os.environ.get('GREENS_CODE_XML')
    XML = os.path.join(greens_path, "input_periodic_cell.xml")

    x = np.linspace(0, L, 20)
    y = np.linspace(0, 1., (pphw*N+1))

    wg_kwargs = {'N': N,
                 'eta': eta,
                 'L': L,
                 'init_phase': init_phase,
                 'loop_direction': loop_direction,
                 'loop_type': loop_type}
    WG = Waveguide(**wg_kwargs)
    WG.x_EP = eps
    eps, delta = WG.get_cycle_parameters(x)

    Bloch_data = []
    Bloch_vector_0 = []
    Bloch_vector_1 = []
    for n, (xn, epsn, deltan) in enumerate(zip(x, eps, delta)):
        print "xn, n", xn, n

        # prepare waveguide and profile
        XML = os.path.join(greens_path, XML)

        profile_kwargs = {'eps': epsn,
                          'delta': deltan,
                          'pphw': pphw,
                          'input_xml': XML,
                          'custom_directory': os.getcwd(),
                          'neumann': 1}
        wg_kwargs_n = {'N': N,
                       'eta': eta,
                       'L': 2*np.pi/(WG.kr + deltan),
                       # 'init_phase': init_phase,
                       'loop_direction': loop_direction,
                       'loop_type': 'Constant'}

        profile_kwargs.update(wg_kwargs_n)
        ep.profile.Generate_Profiles(**profile_kwargs)

        # run code
        cmd = "mpirun -np 4 solve_xml_mumps dev"
        greens_code = subprocess.Popen(cmd.split(),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
        greens_code.communicate()

        for file in glob.glob("N_*profile"):
            if "lower" in file:
                shutil.move(file, "xn_{:08.4f}.lower_profile".format(xn))
            if "upper" in file:
                shutil.move(file, "xn_{:08.4f}.upper_profile".format(xn))

        shutil.move("upper.dat", "xn_{:08.4f}.upper_dat".format(xn))
        shutil.move("lower.dat", "xn_{:08.4f}.lower_dat".format(xn))

        # get Bloch eigensystem
        K, _, ev, _ = bloch.get_eigensystem(return_eigenvectors=True, verbose=False)
        K0, K1 = K[0], K[1]
        ev0, ev1 = ev[0, :], ev[1, :]

        diff_0 = np.abs(np.diff(ev0)).sum()
        diff_1 = np.abs(np.diff(ev1)).sum()
        print 
        print 'diff before', diff_0
        print 'diff before', diff_1
        print 
        print L/2. + init_phase/(WG.kr + deltan)
        # if diff_0 > diff_1 and xn < L/2.:
        if K0 > K1 and xn < L/2.:
            ev0, ev1 = ev1, ev0
            K0, K1 = K1, K0
        # elif diff_0 < diff_1 and xn > L/2.:
        elif K0 < K1 and xn > L/2.:
            ev0, ev1 = ev1, ev0
            K0, K1 = K1, K0


        diff_0 = np.abs(np.diff(ev0)).sum()
        diff_1 = np.abs(np.diff(ev1)).sum()
        print 
        print 'diff after', diff_0
        print 'diff after', diff_1
        print 

        Bloch_data.append([xn, epsn, deltan,
                           K0, K1, ev0, ev1])
        Bloch_vector_0.append(ev0)
        Bloch_vector_1.append(ev1)
        shutil.copyfile("pic.geometry.sine_boundary.1.jpg",
                        "xn_{:08.4f}.jpg".format(xn))
        print xn, epsn, deltan, K0, K1

        z = ev0.view(dtype=float)
        np.savetxt("xn_{:08.4f}.dat".format(xn), zip(y,
                                                     ev0.real,
                                                     ev0.imag,
                                                     np.ones_like(z)*K0.real,
                                                     np.ones_like(z)*K0.imag,
                                                     ev1.real,
                                                     ev1.imag,
                                                     np.ones_like(z)*K1.real,
                                                     np.ones_like(z)*K1.imag),
                   header='y Re(ev0) Im(ev0) Re(K0) Im(K0) Re(ev1) Im(ev1) Re(K1) Im(K1)')

    X, Y = np.meshgrid(x, y)

    Chi_0 = np.array(Bloch_vector_0).T
    Chi_1 = np.array(Bloch_vector_1).T

    plt.pcolormesh(X, Y, np.real(Chi_0 * np.exp(1j*K0*xn)))
    plt.savefig("test_0.png")

    plt.pcolormesh(X, Y, np.real(Chi_1 * np.exp(1j*K1*xn)))
    plt.savefig("test_1.png")


if __name__ == '__main__':
    argh.dispatch_command(get_loop_eigenfunction)
