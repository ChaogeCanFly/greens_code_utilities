#!/usr/bin/env python2.7

import numpy as np
import scipy.linalg

import argparse
from argparse import ArgumentDefaultsHelpFormatter as default_help

from S_Matrix import S_Matrix


class Time_Delay_Matrix(object):
    """Build the time-delay operator Q and calculate the eigensystem of Q11.

        Parameters:
        -----------
            infile: str
                Input file to read S-matrix from.
            coeff_file: str
                Time delay eigenstates output file.
            evals_file: str
                Time delay eigenvalues output file.
            derivative_stepsize: float
                Since the Smat.*.dat file does generally not contain the
                scheduler variable values (except if the energy is varied),
                the derivative-step dP in

                    (S(P0+dP) - S(P0-dP) ) / (2*dP)

                has to be supplied manually. Apart from dP, the S-matrix at
                different values S(P_n) is read from the Smat.*.dat.

        Attributes:
        -----------
            S: S_Matrix object
                The S-matrix of the scattering problem. Since we need to
                discretize the energy derivative numerically, 3 distinct values
                S(E0-dE), S(E0) and S(E0+dE) are required.
            Q: (S.ndim, S.ndim) ndarray
                Time-delay matrix.
            eigenvalues: (S.modes,) ndarray
                Q eigenvalues.
            eigenstates: (S.modes, S.modes) ndarray
                Q eigenstates.
    """

    def __init__(self, infile=None, coeff_file=None, evals_file=None,
                 derivative_stepsize=None):

        S = S_Matrix(infile=infile)
        S0, S1, S2 = [ S.S[n,...] for n in 0, 1, 2 ]
        self.S1 = S1

        modes = S.modes
        self.S = S
        self.modes = modes

        if derivative_stepsize is None:
            dE = np.diff(S.E)[-1]
        else:
            dE = derivative_stepsize

        self.Q = -1j*S1.conj().T.dot(S2-S0)/(2.*dE)
        self.Q11 = self.Q[:modes, :modes]
        self.Q21 = self.Q[modes:, :modes]

        eigenvalues, eigenstates = scipy.linalg.eig(self.Q11)
        idx = eigenvalues.argsort()
        self.eigenvalues = eigenvalues[idx]
        self.eigenstates = eigenstates[:, idx]

        # transmission eigenvalues
        self.t = self.S1[self.modes:, :self.modes]
        self.T = (abs(self.t.dot(self.eigenstates))**2).sum(axis=0)

        # reflection eigenvalues
        self.r = self.S1[:self.modes, :self.modes]
        self.R = (abs(self.r.dot(self.eigenstates))**2).sum(axis=0)

        # not valid if time-reversal symmetry is violated?
        chi = self.Q21.dot(self.eigenstates)
        nullspace_norm =  np.linalg.norm(chi, axis=0)

        self.coeff_file = coeff_file
        self.evals_file = evals_file
        self.nullspace_norm = nullspace_norm

    def write_eigenstates(self):
        """Write the coefficients of the time-delay-operator eigenstate in a
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
            self.coeff_file = 'coeff.Q_states.dat'

        with open(self.coeff_file, "w") as f:
            f.write('1 -1\n')
            f.write('1.0 0.5\n')
            f.write('{} {}\n'.format(self.S.E[1], self.modes))

            for n in range(self.modes):
                f.write('\n')
                f.write('1.0\n')
                for m in range(self.modes):
                    v = self.eigenstates[m,n]
                    f.write('({v.real}, {v.imag})\n'.format(v=v))

    def write_eigenvalues(self):
        """Write the eigenvalues of the time-delay-operator."""

        if not self.evals_file:
            self.evals_file = 'evals.Q_states.dat'

        np.savetxt(self.evals_file, zip(self.eigenvalues, self.T, self.R,
                                        self.T + self.R, self.nullspace_norm),
                   header=('delay times q, transmission probabilities '
                           'T = |t q|**2 and R = |r q|**2, T+R, '
                           'and nullspace norms |Q21 q|'),
                   fmt='%.8e')


def parse_arguments():
    """Parse command-line arguments and write the Q_matrix eigenstates."""

    parser = argparse.ArgumentParser(formatter_class=default_help)

    parser.add_argument("-i", "--infile", default=None,
                        type=str, help="Input file to read S-matrix from.")
    parser.add_argument("-c", "--coeff-file", default='coeff.Q_states.dat',
                        type=str, help="Time delay eigenstates output file.")
    parser.add_argument("-e", "--evals-file", default='evals.Q_states.dat',
                        type=str, help="Time delay eigenvalues output file.")
    parser.add_argument("-d", "--derivative-stepsize", default=None, type=float,
                        help=("Derivative-step for which to calculate"
                              "the Q-matrix."))

    parse_args = parser.parse_args()
    args = vars(parse_args)

    Q = Time_Delay_Matrix(**args)
    Q.write_eigenstates()
    Q.write_eigenvalues()


if __name__ == '__main__':
    parse_arguments()
