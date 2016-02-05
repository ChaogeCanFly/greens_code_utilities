#!/usr/bin/env python2.7

import glob
import numpy as np

import argh

from helper_functions import convert_to_complex
from xmlparser import XML


def get_eigensystem(xml='input.xml', evalsfile=None, evecsfile=None,
                    modes=None, L=None, dx=None, r_nx=None, sort=True,
                    return_velocities=False, return_eigenvectors=False,
                    verbose=True, neumann=False):
    """Extract the eigenvalues beta and return the Bloch modes.

        Parameters:
        -----------
            xml: str
                Input xml file.
            evalsfile: str
                Eigenvalues input file. Defaults to first element in working
                directory matching Evals.*.dat.
            evecsfile: str
                Eigenvectors input file. Defaults to first element in working
                directory matching Evecs.*.dat.
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
            return_velocities: bool
                Whether to return eigenvalues and group velocities.
            return_eigenvectors: bool
                Whether to return eigenvalues and eigenvectors.
            verbose: bool
                Print additional output.
            neumann: bool
                Whether to use Neumann or Dirichlet boundary conditions.

        Returns:
        --------
            k_left, k_right: (N,) ndarrays
                Bloch modes of left and right movers.
            chi_left, chi_right: (N,N) ndarrays (optional)
                Eigenvectors of left and right movers.
            v_left, v_right: (N,) ndarrays (optional)
                Velocities of left and right movers.
    """

    if evalsfile is None:
        evalsfile = glob.glob("Evals.*.dat")
    if len(evalsfile) != 1:
        print """Warning: found multiple files matching Evals.*.dat!
                 Proceeding with file {}.""".format(evalsfile[0])
        evalsfile = evalsfile[0]

    if modes is None or dx is None or r_nx is None:
        if verbose:
            print "# Parameters 'modes', 'dx' and 'r_nx' not found."
            print "# Reading xml file {}.".format(xml)
        xml_params = XML(xml).params
        modes = xml_params.get("modes")
        L = xml_params.get("L")
        dx = xml_params.get("dx")
        r_nx = xml_params.get("r_nx")

    # get k_x values for both modes
    if neumann:
        k0, k1 = [np.sqrt(modes**2 - n**2)*np.pi for n in (0, 1)]
    else:
        k0, k1 = [np.sqrt(modes**2 - n**2)*np.pi for n in (1, 2)]
    kr = k0 - k1

    # get reciprocal lattice vector G
    G = 2.*np.pi/L

    # get eigenvectors chi_n (performance critical!)
    if return_eigenvectors:
        if evecsfile is None:
            evecsfile = glob.glob("Evals.*.dat")
        if len(evecsfile) != 1:
            print """Warning: found multiple files matching Evals.*.dat!
                    Proceeding with file {}.""".format(evecsfile[0])
            evecsfile = evecsfile[0]
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
    # k /= dx*r_nx
    k /= L  # = period L for chi(x+L) = chi(x) (note that generally L!=r_nx*dx)

    # --- experimental: sort the array according to velocities first, then fill
    # v_left and v_right with left and rightmovers ----------------------------
    # initial_sort = np.argsort(velocities.real)
    # k = k[initial_sort]
    # velocities = velocities[initial_sort]
    # take care that other subroutines now use the correct order of k_l, k_r
    # -------------------------------------------------------------------------

    k_left = k[:len(k)/2]
    k_right = k[len(k)/2:]

    v_left = velocities[:len(k)/2]
    v_right = velocities[len(k)/2:]

    # --- experimental: len(v_left) != len(v_right) ---------------------------
    # v_left = velocities[velocities.real < 0]
    # v_right = velocities[velocities.real > 0]
    # k_left = k[velocities.real < 0]
    # k_right = k[velocities.real > 0]
    # -------------------------------------------------------------------------

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

    # the following procedure is redundant:
    # np.angle(x) maps x into the domain [-pi,pi], i.e.,
    #   -pi <= k*L <= pi,
    # thus the resulting eigenvalue is already given in the 1. BZ:
    #   -pi/L <= k <= pi/L
    #
    # if fold_back:
    #     # k_left, k_right = [ np.mod(x.real, kr) + 1j*x.imag
    #     k_left, k_right = [ np.mod(x.real, G) + 1j*x.imag
    #                          for x in k_left, k_right ]
    #     # map into 1.BZ
    #     for k in k_left, k_right:
    #         k[k > G/2] -= G
    #         k[k < -G/2] += G

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
        print "|G - kr|:", abs(G - kr)

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
