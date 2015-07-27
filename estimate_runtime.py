#!/usr/bin/env python

from __future__ import division

import argparse
from argparse import RawDescriptionHelpFormatter
import math
import sys


def get_runtime(pphw=None, modes=None, length=None, width=None, nx=None,
                ny=None, Ncores=None, machine=None):
    """Print the estimated greens_code runtime.

    If neither nx nor ny are provided, the variables pphw, modes, length and
    width have to be supplied to determine the grid-spacing. If Ncores is
    unspecified, the estimated runtimes for 2^3, ..., 2^9 cores are printed.

        Parameters:
        -----------
            pphw: int
                Points per half wavelength.
            modes: float
                Number of open modes (floor(N)).
            length: float
                System length.
            width: float
                System width.
            nx: int
                Number of grid-points in longitudinal direction.
            ny: int
                Number of grid-points in transversal direction.
            Ncores: int
                Number of cores.
            machine: str
                Machine on which greens_code is run (VSC2|VSC3).

        Returns:
        --------
            T: float
                Estimated runtime in seconds.

        Notes:
        ------
            Valid for greens_code revision 482 and above (as long as no
            performance-critical changes are made).
    """

    if not nx or not ny:
        nyout = pphw*modes
        dy = width/(nyout + 1)
        dx = dy
        nx = int(length/dx)
        ny = int(width/dy)

    if machine == 'VSC2':
        print ("Warning: VSC2 parameters have not yet been determined. Using "
               "VSC3 predictions in the following.")
        a = 6.
        b = 3.5
    elif machine == 'VSC3':
        a = 6.
        b = 3.5

    T = (a*ny**3*(nx/Ncores) + math.log(Ncores, 2)*b*ny**3)*1e-9

    T_h = T // 3600
    T_m = (T // 60) % 60
    T_s = T % 60

    print "Estimated runtime for {:4} core(s): {:.0f}h {:2.0f}m {:2.0f}s".format(Ncores, T_h, T_m, T_s)

    return T


def parse_arguments():
    """Parse command-line arguments and call get_runtime()."""

    parser = argparse.ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                                     description=get_runtime.__doc__)

    parser.add_argument("-p", "--pphw", default=None, type=int)
    parser.add_argument("-m", "--modes", default=None, type=float)
    parser.add_argument("-l", "--length", default=None, type=float)
    parser.add_argument("-w", "--width", default=None, type=float)
    parser.add_argument("-x", "--nx", default=None, type=int)
    parser.add_argument("-y", "--ny", default=None, type=int)
    parser.add_argument("-N", "--Ncores", default=None, type=int)
    parser.add_argument("-M", "--machine", default='VSC3', type=str)

    parse_args = parser.parse_args()
    args = vars(parse_args)

    if not args.get("Ncores"):
        for n in range(3, 10):
            N = 2**n
            args['Ncores'] = N
            get_runtime(**args)
    else:
        get_runtime(**args)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print "No command line arguments found. Use estimate_runtime.py -h for help"
    else:
        parse_arguments()
