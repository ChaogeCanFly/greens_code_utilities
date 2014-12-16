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
            oufile: str
                Output coefficients file.

        Attributes:
        -----------
            S: S_Matrix object
                The S-matrix of the scattering problem. Since we need to
                discretize the energy derivative numerically, 3 distinct values
                S(E0), S(E0+dE) and S(E0+2dE) are required.
            Q: (S.ndim, S.ndim) ndarray
                Time-delay matrix.
    """

    def __init__(self, outfile=None, infile=None):

        S = S_Matrix(infile=infile)
        S0, S1, S2 = [ S.S[n,...] for n in 0, 1, 2 ]
        dE = np.diff(S.E)[-1]
        modes = S.modes
        self.S = S
        self.modes = modes

        self.Q = -1j*S1.conj().T.dot(S2-S0)/(2.*dE)
        self.Q11 = self.Q[:modes, :modes]
        self.Q21 = self.Q[modes:, :modes]

        delay_times, delay_eigenstates = scipy.linalg.eig(self.Q11)

        self.delay_times = delay_times
        self.delay_eigenstates = delay_eigenstates.T

        self.outfile = outfile

    def write_states(self):
        """Write the coefficients of the time-delay-operator eigenstate in a
        format readable by the greens_code:

            1 -1        # number of datasets; which state to plot
            1.0 0.5     # definition of ID corridors: plot states ranging
                          from -0.5 to 1.5
            E modes     # scattering energy; number of open modes

            ID_0
            coefficients_0

            etc.
        """
        if not self.outfile:
            self.outfile = 'coeff.Q_states.dat'

        with open(self.outfile, "w") as f:
            f.write('1 -1\n')
            f.write('1.0 0.5\n')
            f.write('{} {}\n'.format(self.S.E[1], self.modes))

            for n in range(self.modes):
                f.write('\n')
                f.write('1.0\n')
                for m in range(self.modes):
                    v = self.delay_eigenstates[n,m]
                    f.write('({v.real}, {v.imag})\n'.format(v=v))


def parse_arguments():
    """Parse command-line arguments and write the Q_matrix eigenstates."""

    parser = argparse.ArgumentParser(formatter_class=default_help)

    parser.add_argument("-i", "--infile", default=None,
                        type=str, help="Input file to read S-matrix from.")
    parser.add_argument("-o", "--outfile", default='coeff.Q_states.dat',
                        type=str, help="Time delay eigenstates output file.")

    parse_args = parser.parse_args()
    args = vars(parse_args)

    Q = Time_Delay_Matrix(**args)
    Q.write_states()


if __name__ == '__main__':
    parse_arguments()
