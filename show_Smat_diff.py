#!/usr/bin/env python2.7

import sys 
import numpy as np

def show_Smat_diff(a, b):
    """Print difference between input files a and b."""

    params = {'unpack': True, 
              'skiprows': 4}

    a, b = [ np.loadtxt(i, **params) for i in (a,b) ]

    print "File a:" 
    print a
    print
    print "File b:"
    print b
    print
    print "Difference (a - b):"
    print a - b

if __name__ == '__main__':
    show_Smat_diff(sys.argv[1], sys.argv[2])
