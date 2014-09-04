#!/usr/bin/env python2.7

import numpy as np
import re

import argh

from helpers import convert_to_complex
from xmlparser import XML


def get_eigenvalues(xml='input.xml', evalsfile='Evals.sine_boundary.dat',
                    evecsfile='Evecs.sine_boundary.dat',
                    modes=None, L=None, dx=None, r_nx=None, sort=True,
                    fold_back=True, return_velocities=False, return_all=False,
                    verbose=True):
    """Extract the eigenvalues beta and return the Bloch modes.

        Parameters:
        -----------
            xml: str
                Input xml file.
            evalsfile: str
                Eigenvalues input file.
            evecsfile: str
                Eigenvectors input file.
            modes: float
                Number of open modes.
            L: float
                System length.
            dx: float
                Grid spacing.
            r_nx: int
                Grid dimension in x-direction.
            fold_back: bool
                Whether to fold back the Bloch modes into the 1. BZ.
            return_velocities: bool
                Whether to return group velocities.
            return_all: bool
                Whether to return eigenvalues and eigenvectors.
            verbose: bool
                Print additional output.

        Returns:
        --------
            k_left, k_right: ndarrays
                Bloch modes of left and right movers.
            v_left, v_right: ndarrays (optional)
                Velocities of left and right movers.
    """

    if modes is None or dx is None or r_nx is None:
        params = XML(xml).params
        modes = params.get("modes")
        L = params.get("L")
        dx = params.get("dx")
        r_nx = params.get("r_nx")

    # get k_x values for both modes
    k0, k1 = [ np.sqrt(modes**2 - n**2)*np.pi for n in (0, 1) ]
    kr = k0 - k1

    # reciprocal lattice vector
    G = 2.*np.pi/L

    # get eigenvectors
    chi = np.loadtxt(evecsfile, dtype=str)
    chi = [ map(convert_to_complex, x) for x in chi ]
    chi = np.asarray(chi)
    chi_left = chi[:len(chi)//2]
    chi_right = chi[:len(chi)//2]

    # get beta = exp(ik_x*dx) and group velocities
    beta, velocities = np.genfromtxt(evalsfile, unpack=True,
                                     usecols=(0, 1), dtype=complex,
                                     converters={0: convert_to_complex,
                                                 1: convert_to_complex})
    k = np.angle(beta) - 1j*np.log(np.abs(beta))
    k /= dx*r_nx
    k_left = k[:len(k)/2]
    k_right = k[len(k)/2:]

    v_left = velocities[:len(k)/2]
    v_right = velocities[len(k)/2:]

    if sort:
        sort_left = np.argsort(abs(k_left.imag))
        k_left = k_left[sort_left]
        v_left = v_left[sort_left]
        chi_left = chi_left[sort_left]

        sort_right = np.argsort(abs(k_right.imag))
        k_right = k_right[sort_right]
        v_right = v_right[sort_right]
        chi_right = chi_right[sort_right]

    if fold_back:
        k_left, k_right = [ np.mod(x.real, kr) + 1j*x.imag for x in 
                                                             k_left, k_right ]
        # map into 1.BZ
        for k in k_left, k_right:
            k[k > G/2] -= G
            k[k < -G/2] += G

    if verbose:
        print "modes", modes
        print "L", L
        print "G", G
        print "G/2", G/2
        print "k0", k0
        print "k1", k1
        print "kr", kr
        print "dx:", dx
        print "r_nx:", r_nx
        print "dx*r_nx:", dx*r_nx

    if return_velocities:
        return k_left, k_right, v_left, v_right
    elif return_all:
        return k_left, k_right, chi_left, chi_right
    else:
        return k_left, k_right


if __name__ == '__main__':
    argh.dispatch_command(get_eigenvalues)
