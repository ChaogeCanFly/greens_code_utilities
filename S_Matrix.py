#!/usr/bin/env python
# TODO:
#  * write different nruns into same line or in different lines?
#    -> in _process_directories()

import glob
import numpy as np
import os

import argparse
from argparse import ArgumentDefaultsHelpFormatter as default_help

# remove numpy's conversion warnings ------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=np.lib._iotools.ConversionWarning)
# -----------------------------------------------------------------------------


class S_Matrix(object):
    """Reads and processes the S-matrix.

            Parameters:
            -----------
                indir: str
                    Input directory.
                infile: str
                    Input file to read S-matrix from.
                probabilities: bool
                    Whether to calculate abs(S)^2.
                from_right: bool
                    Whether to use the S-matrix for injection from right.
    """

    def __init__(self, infile=None, indir=".", probabilities=False,
                 from_right=False):

        self.indir = indir
        if not infile:
            try:
                infile = glob.glob("{}/Smat.*.dat*".format(indir))
                if len(infile) > 1:
                    print "Warning: more than one Smat.*.dat* found:"
                    for smat in infile:
                        print "\t" + os.path.basename(smat)
                    print "Using the first entry in the following..."
                self.infile = infile[0]
            except:
                pass
        else:
            self.infile = os.path.join(indir, infile)

        self.probabilities = probabilities
        self.from_right = from_right
        self._get_amplitudes()

    def _get_amplitudes(self):
        """Get transmission and reflection amplitudes, as well as the S-matrix
        dimensions, number of runs and the scattering energies."""

        try:
            # get number of scheduler steps and S-matrix dimensions
            nruns, ndims = np.genfromtxt(self.infile,
                                         invalid_raise=False)[:3:2]
        except:
            print "Warning: couldn't determine S-matrix dimensions."
            nruns, ndims = (1, 4)

        try:
            # get real and imaginary parts of S-matrix
            re, im = np.genfromtxt(self.infile, usecols=(2, 3), autostrip=True,
                                   unpack=True, invalid_raise=False)
            # calculate transmission and reflection amplitudes
            S = (re + 1j*im).reshape((nruns, ndims, ndims))
        except:
            # initialize nan S-matrix if no data available
            S = np.empty((nruns, ndims, ndims))
            S[:] = np.nan

        try:
            # get scattering energies (k**2/2.)
            E = np.genfromtxt(self.infile, usecols=(0), autostrip=True,
                              unpack=True, invalid_raise=False)
            self.E = E[1::ndims*ndims+2]
        except:
            pass

        if self.probabilities:
            self.S_amplitudes = S
            self.S = abs(S)**2
        else:
            self.S_amplitudes = S
            self.S = S

        self.nruns = int(nruns)
        self.ndims = int(ndims)
        self.modes = int(ndims) // 2

        if self.from_right:
            tmp = 1.*self.S
            N = self.modes
            # r_nm -> r'_nm
            self.S[:, :N, :N] = tmp[:, N:, N:]
            # t_nm -> t'_nm
            self.S[:, N:, :N] = tmp[:, :N, N:]
            # r'_nm -> r_nm
            self.S[:, N:, N:] = tmp[:, :N, :N]
            # t'_nm -> t_nm
            self.S[:, :N, N:] = tmp[:, N:, :N]


def natural_sorting(text, args="delta", delimiter="_"):
    """Sort text with respect to the argument value.

        Parameters:
        -----------
            text: str
                String to be sorted.
            args: str
                Directory parsing parameters.
    """
    index = lambda text: [text.split(delimiter).index(arg) for arg in args]
    alphanum_key = lambda text: [float(text.split(delimiter)[i+1]) for
                                 i in index(text)]

    return sorted(text, key=alphanum_key)


class Write_S_Matrix(object):
    """Class which handles directories and globbing.

        Parameters:
        -----------
            infile: str
                Input file to read S-matrix from.
            probabilities: bool
                Whether to calculate abs(S)^2.
            outfile: str
                S-matrix output file.
            directories: list of str
                Directories to parse.
            glob_args: list of str
                Directory parsing parameters.
            delimiter: str
                Directory parsing delimiter.
            from_right: bool
                 Whether to use the S-matrix for injection from right.
            full_smatrix: bool
                 Whether to return the full S-matrix.
            total_probabilities: bool
                 Whether to return the total mode probabilities.
    """

    def __init__(self, infile=None, probabilities=False,
                 outfile="S_matrix.dat", directories=[], glob_args=[],
                 delimiter="_", from_right=False, full_smatrix=False,
                 total_probabilities=False, **s_matrix_kwargs):

        self.outfile = outfile
        self.directories = directories
        self.glob_args = glob_args
        self.nargs = len(glob_args)
        self.delimiter = delimiter
        self.probabilities = probabilities
        self.full_smatrix = full_smatrix
        self.total_probabilities = total_probabilities
        self.s_matrix_kwargs = {'infile': infile,
                                'probabilities': probabilities,
                                'from_right': from_right}

        self._process_directories()

    def _process_directories(self):
        """Loop through all directories satisfying the globbing pattern or the
        supplied list of directories."""

        if self.directories:
            dirs = self.directories
        elif self.glob_args:
            dirs = sorted(glob.glob("*" + self.glob_args[0] + "*"))
        else:
            dirs = [os.getcwd()]

        dirs = natural_sorting(dirs, args=self.glob_args,
                               delimiter=self.delimiter)

        with open(self.outfile, "w") as f:
            for (n, dir) in enumerate(dirs):
                self.S = S_Matrix(indir=dir, **self.s_matrix_kwargs)
                if not n:
                    f.write(self._get_header(dir) + "\n")
                f.write(self._get_data(dir) + "\n")

    def _parse_directory(self, dir):
        """Extract running variables from directory name."""

        dir = dir.split(self.delimiter)
        arg_values = []

        try:
            for p in self.glob_args:
                idx = dir.index(p)
                arg_values.append(dir[idx+1])
        except:
            arg_values.append([])

        return arg_values

    def _get_header(self, dir):
        """Prepare data file header."""

        S = self.S

        # tune alignment spacing
        spacing = 17 if S.probabilities else 35

        # translate S-matrix into transmission and reflection components
        if self.probabilities:
            header_variables = ("R", "T")
            header_prime_variables = ("T'", "R'")
        else:
            header_variables = ("r", "t")
            header_prime_variables = ("t'", "r'")

        header = ["{0}{1}{2}".format(s, i, j)
                  for s in header_variables
                  for i in range(S.modes)
                  for j in range(S.modes)]

        if self.full_smatrix:
            header_prime = ["{0}{1}{2}".format(s, i, j)
                            for s in header_prime_variables
                            for i in range(S.modes)
                            for j in range(S.modes)]
            header += header_prime

        if self.total_probabilities:
            header_total = ["{0}{1}".format(s, i, j)
                            for s in header_variables
                            for i in range(S.modes)]
            header_total += header_variables
            header += header_total

        headerfmt = '#'
        headerfmt += "  ".join(['{:>12}' for n in range(self.nargs)]) + "  "
        headerfmt += "  ".join(['{:>{s}}' for n in range(len(header))])

        header = headerfmt.format(*(self.glob_args + header), s=spacing)

        return header

    def _get_data(self, dir):
        """Prepare S-matrix data for output."""

        arg_values = self._parse_directory(dir)
        S = self.S

        # write r and t (dimension 2modes*modes)
        data = [S.S[i, j, k]
                for i in range(S.nruns)
                for j in range(S.ndims)
                for k in range(S.modes)]

        # join data
        if self.full_smatrix:
            # write t' and r' (dimension 2modes*modes)
            data_prime = [S.S[i, j, k + S.modes]
                          for i in range(S.nruns)
                          for j in range(S.ndims)
                          for k in range(S.modes)]
            data += data_prime

        if self.total_probabilities:
            data_total = [[np.abs(S.S_amplitudes[0, i, j])**2
                           for i in range(S.ndims)]
                          for j in range(S.modes)]
            # sum over columns
            data_total = np.sum(data_total, axis=0)
            T_total = np.sum(data_total[S.modes:])
            R_total = np.sum(data_total[:S.modes])
            data += data_total.tolist() + [R_total] + [T_total]

        datafmt = " "
        datafmt += "  ".join(['{:>12}' for n in range(self.nargs)]) + "  "
        datafmt += "  ".join(['{:> .10e}' for n in range(len(data))])

        data = datafmt.format(*(arg_values + data))

        return data


def get_S_matrix_difference(a, b):
    """Print differences between input files.

        Parameters:
        -----------
            a, b: str
                Relative paths of S-matrix input files.
    """

    params = {'unpack': True,
              'skiprows': 4}

    a, b = [np.loadtxt(i, **params) for i in (a, b)]

    print "File a:"
    print a
    print
    print "File b:"
    print b
    print
    print "Difference (a - b):"
    print a - b


def test_S_matrix_symmetry():
    """Test if S-matrix is transposition symmetric, S^T = S."""
    S = abs(S_Matrix().S[0])**2
    ST = abs(S_Matrix().S[0].T)**2
    print "|S|^2:"
    print S
    print
    print "|S|^2 - |S^T|^2:"
    print S - ST


def parse_arguments():
    """Parse command-line arguments and call Write_S_matrix."""

    parser = argparse.ArgumentParser(formatter_class=default_help)

    parser.add_argument("-i", "--infile", default=None,
                        type=str, help="Input file to read S-matrix from.")
    parser.add_argument("-p", "--probabilities", action="store_true",
                        help="Whether to calculate abs(S)^2.")
    parser.add_argument("-f", "--full-smatrix", action="store_true",
                        help=("Whether to write the full S-matrix (including "
                              "the primed matrices t' and r)'."))
    parser.add_argument("-t", "--total-probabilities",
                        action="store_true",
                        help=("Whether to add the total mode transmission and "
                              "reflection to the output file."))
    parser.add_argument("-d", "--directories", default=[], nargs="*",
                        help="Directories to parse.")
    parser.add_argument("-g", "--glob-args", default=[], nargs="*",
                        help="Directory parsing variables.")
    parser.add_argument("-l", "--delimiter", default="_",
                        type=str, help="Directory parsing delimiters.")
    parser.add_argument("-r", "--from-right", action="store_true",
                        help=("Whether to use the S-matrix for injection "
                              "from right."))
    parser.add_argument("-o", "--outfile", default="S_matrix.dat",
                        type=str, help="S-matrix output file.")

    parser.add_argument("-D", "--diff", default=[], nargs="*",
                        help="Print difference between input files.")

    parser.add_argument("-S", "--symmetric", action="store_true",
                        help="Test if S-matrix is transposition symmetric.")

    parse_args = parser.parse_args()
    args = vars(parse_args)

    if parse_args.diff:
        get_S_matrix_difference(*parse_args.diff)
    else:
        del args['diff']
        Write_S_Matrix(**args)

    if parse_args.symmetric:
        test_S_matrix_symmetry()


if __name__ == '__main__':
    parse_arguments()
