#!/usr/bin/env python2.7

import numpy as np
import re

import argh

from helpers import convert_to_complex
from xmlparser import XML


def get_eigensystem(xml='input.xml', evalsfile='Evals.sine_boundary.dat',
                    evecsfile='Evecs.sine_boundary.dat', modes=None, L=None,
                    dx=None, r_nx=None, sort=True, fold_back=True, return_velocities=False,
                    return_eigenvectors=False, verbose=True):
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
            sort: bool
                Whether to sort the eigenvalues and eigenvectors.
            fold_back: bool
                Whether to fold back the Bloch modes into the 1. BZ.
            return_velocities: bool
                Whether to return eigenvalues and group velocities.
            return_eigenvectors: bool
                Whether to return eigenvalues and eigenvectors.
            verbose: bool
                Print additional output.

        Returns:
        --------
            k_left, k_right: (N,) ndarrays
                Bloch modes of left and right movers.
            chi_left, chi_right: (N,N) ndarrays (optional)
                Eigenvectors of left and right movers.
            v_left, v_right: (N,) ndarrays (optional)
                Velocities of left and right movers.
    """

    if modes is None or dx is None or r_nx is None:
        if verbose:
            print "# Parameters 'modes', 'dx' and 'r_nx' not found."
            print "# Reading xml file {}.".format(xml)
        params = XML(xml).params
        modes = params.get("modes")
        L = params.get("L")
        dx = params.get("dx")
        r_nx = params.get("r_nx")

    # get k_x values for both modes
    k0, k1 = [ np.sqrt(modes**2 - n**2)*np.pi for n in (0, 1) ]
    kr = k0 - k1

    # get reciprocal lattice vector G
    G = 2.*np.pi/L

    # get eigenvectors chi_n
    if return_eigenvectors:
        chi = np.loadtxt(evecsfile, dtype=str)
        chi = [ map(convert_to_complex, x) for x in chi ]
        chi = np.asarray(chi)

        chi_left = chi[:len(chi)//2]
        chi_right = chi[len(chi)//2:]

    # get beta = exp(i*K_n*dx) and group velocities v_n
    beta, velocities = np.genfromtxt(evalsfile, unpack=True,
                                     usecols=(0, 1), dtype=complex,
                                     converters={0: convert_to_complex,
                                                 1: convert_to_complex})
    k = np.angle(beta) - 1j*np.log(np.abs(beta))
    k /= dx*r_nx

    # --- experimental: sort the array according to velocities first, then
    # fill v_left and v_right with left and rightmovers
    # initial_sort = np.argsort(velocities.real)
    # k = k[initial_sort]
    # velocities = velocities[initial_sort]
    # take care that other subroutines now use the correct order of k_l, k_r
    # ---

    k_left = k[:len(k)/2]
    k_right = k[len(k)/2:]

    v_left = velocities[:len(k)/2]
    v_right = velocities[len(k)/2:]

    # --- experimental: len(v_left) != len(v_right)
    # v_left = velocities[velocities.real < 0]
    # v_right = velocities[velocities.real > 0]
    # k_left = k[velocities.real < 0]
    # k_right = k[velocities.real > 0]
    # ---

    if sort:
        sort_right = np.argsort(abs(k_right.imag))
        k_right = k_right[sort_right]
        v_right = v_right[sort_right]

        sort_left = np.argsort(abs(k_left.imag))
        k_left = k_left[sort_left]
        v_left = v_left[sort_left]

        if return_eigenvectors:
            chi_left = chi_left[sort_left]
            chi_right = chi_right[sort_right]

        # TODO: handle conservative case


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
        print "|L - dx*r_nx|:", abs(L - dx*r_nx)

    if return_velocities and not return_eigenvectors:
        return k_left, k_right, v_left, v_right
    elif return_eigenvectors and not return_velocities:
        return k_left, k_right, chi_left, chi_right
    elif return_eigenvectors and return_velocities:
        return k_left, k_right, chi_left, chi_right, v_left, v_right
    else:
        return k_left, k_right


if __name__ == '__main__':
    argh.dispatch_command(get_eigensystem)
