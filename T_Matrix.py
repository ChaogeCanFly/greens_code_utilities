#!/usr/bin/env python2.7

import numpy as np
import scipy.linalg

import argparse
from argparse import ArgumentDefaultsHelpFormatter as default_help

from S_Matrix import S_Matrix


class T_Matrix(object):
    """Build the transmission probability operator t*^T t and calculate the
    eigensystem.

        Parameters:
        -----------
            infile: str
                Input file to read S-matrix from.
            coeff_file: str
                Eigenstates output file.
            evals_file: str
                Eigenvalues output file.

        Attributes:
        -----------
            S: S_Matrix object
                The S-matrix of the scattering problem.
            T: (S.modes, S.modes) ndarray
                The t*^T t matrix.
            eigenvalues: (S.modes,) ndarray
                t*^T t eigenvalues.
            eigenstates: (S.modes, S.modes) ndarray
                t*^T t eigenstates.
    """

    def __init__(self, infile=None, coeff_file=None, evals_file=None):

        S = S_Matrix(infile=infile)

        modes = S.modes
        self.S = S
        self.modes = modes

        if self.S.nruns == 1:
            self.t = S.S[0, modes:, :modes]
        elif self.S.nruns == 3:
            self.t = S.S[1, modes:, :modes]

        self.T = self.t.conj().T.dot(self.t)

        eigenvalues, eigenstates = scipy.linalg.eig(self.T)
        self.eigenvalues = eigenvalues
        self.eigenstates = eigenstates

        self.coeff_file = coeff_file
        self.evals_file = evals_file

    def write_eigenstates(self):
        """Write the coefficients of the t*^T t operator eigenstate in a
        format readable by the greens_code:

            1 -1        # number of datasets; which state to plot
            1.0 0.5     # definition of ID corridors: plot states ranging
                          from -0.5 to 1.5
            E modes     # scattering energy; number of open modes

            ID_0
            coefficients_0

            ID_1
            coefficients_1

            etc.
        """
        if not self.coeff_file:
            self.coeff_file = 'coeff.T_states.dat'

        if self.S.nruns == 1:
            E = self.S.E[0]
        elif self.S.nruns == 3:
            E = self.S.E[1]

        with open(self.coeff_file, "w") as f:
            f.write('1 -1\n')
            f.write('1.0 0.5\n')
            f.write('{} {}\n'.format(E, self.modes))

            for n in range(self.modes):
                f.write('\n')
                f.write('1.0\n')
                for m in range(self.modes):
                    v = self.eigenstates[m,n]
                    f.write('({v.real}, {v.imag})\n'.format(v=v))

    def write_eigenvalues(self):
        """Write the eigenvalues of the time-delay-operator."""

        if not self.evals_file:
            self.evals_file = 'evals.T_states.dat'

        # readin with np.loadtxt(self.evals_file).view(complex)
        np.savetxt(self.evals_file, zip(self.eigenvalues.real,
                                        self.eigenvalues.imag),
                   header=('t*^T t eigenvalues'),
                   fmt='%.8e')


def parse_arguments():
    """Parse command-line arguments and write the T_matrix eigenstates."""

    parser = argparse.ArgumentParser(formatter_class=default_help)

    parser.add_argument("-i", "--infile", default=None,
                        type=str, help="Input file to read S-matrix from.")
    parser.add_argument("-c", "--coeff-file", default='coeff.T_states.dat',
                        type=str, help="t*^T t eigenstates output file.")
    parser.add_argument("-e", "--evals-file", default='evals.T_states.dat',
                        type=str, help="t*^T t eigenvalues output file.")

    parse_args = parser.parse_args()
    args = vars(parse_args)

    T = T_Matrix(**args)
    T.write_eigenstates()
    T.write_eigenvalues()


if __name__ == '__main__':
    parse_arguments()
