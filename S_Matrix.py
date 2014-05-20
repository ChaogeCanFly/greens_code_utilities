#!/usr/bin/env python2.7

import re
import numpy as np
import argparse
from argparse import ArgumentDefaultsHelpFormatter as default_help
from glob import glob
import os

class S_Matrix:
    """Reads and processes the S-matrix.
            
            Parameters:
            -----------
                indir: str
                    Input directory.
                infile: str
                    Input file to read S-matrix from.
                probabilities: bool
                    Whether to calculate abs(S)^2.
    """

    def __init__(self, indir=".", infile="Smat.sine_boundary.dat", probabilities=False):

        self.indir = indir
        self.infile = indir + "/" + infile
        self.probabilities = probabilities
        self._get_amplitudes()        

    def _get_amplitudes(self):
        """Get transmission and reflection amplitudes."""
        
        try:
            # get number of scheduler steps and S-matrix dimensions
            nruns, ndims = np.loadtxt(self.infile)[:3:2]
        except:
            nruns, ndims = 0, 0

        try:
            # get real and imaginary parts of S-matrix
            re, im = np.genfromtxt(self.infile, usecols=(2,3), autostrip=True,
                                   unpack=True, invalid_raise=False)
            # calculate transmission and reflection amplitudes
            S = (re + 1j*im).reshape((nruns,ndims,ndims))
        except:
            # initialize zero_like S-matrix if no data available
            S = np.zeros((ndims,ndims))
            
        if self.probabilities:
            self.S = abs(S)**2
        else:
            self.S = S
            
        self.nruns = int(nruns)
        self.ndims = int(ndims)    


def natural_sorting(text, glob_arg):
    """Sort text.

        Parameters:
        -----------
            text: str
                String to be sorted.
            glob_arg: 
                Directory parsing parameters.
    """

    glob_arg = glob_arg[0]
    idx = text.index(glob_arg)
    
    return float(text.split("_")[idx+1])

  
class Write_S_Matrix:
    """Class which handles directories and globbing.

        Parameters:
        -----------
            outfile: str
                S-matrix output file.
            glob_args: list of str
                Directory parsing parameters.
            delimiter: str
                Directory parsing delimiter.
    """
    
    def __init__(self, outfile="S_matrix.dat",
                       glob_args=[],
                       delimiter="_", 
                       **kwargs):
                       
        self.outfile = outfile
        self.glob_args = glob_args
        self.nargs = len(glob_args) if glob_args else 0
        self.delimiter = delimiter
        self.kwargs = kwargs

        self._process_directories()

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
        
        S = S_Matrix(indir=dir, **self.kwargs)
        
        # tune alignment spacing
        spacing = 17 if S.probabilities else 35
        
        # translate S-matrix into transmission and reflection components
        header = [ "{}{}{}".format(s,i,j) for s in ("r","t") 
                                          for i in range(S.ndims//2) 
                                          for j in range(S.ndims//2) ]
        headerdim = S.ndims*S.ndims//2  
        
        headerfmt = '#'
        headerfmt += "  ".join([ '{:>12}' for n in range(self.nargs) ]) + "  "     
        headerfmt += "  ".join([ '{:>{s}}' for n in range(headerdim) ])
        
        header = headerfmt.format(*(self.glob_args + header), s=spacing) 
        
        return header   

    def _get_data(self, dir):
        """Prepare S-matrix data for output."""
            
        arg_values = self._parse_directory(dir)    
        S = S_Matrix(indir=dir, **self.kwargs)
        
        data = [ S.S[i,j,k] for i in range(S.nruns) 
                            for j in range(S.ndims)
                            for k in range(S.ndims//2) ]
        datadim = S.ndims*S.ndims//2   
     
        datafmt = " "
        datafmt += "  ".join([ '{:>12}' for n in range(self.nargs) ]) + "  "
        datafmt += "  ".join([ '{:> .10e}' for n in range(datadim) ])
        
        data = datafmt.format(*(arg_values + data))
        
        return data        
  
    def _process_directories(self):
        """Loop through all directories satisfyling the globbing pattern."""

        dirs = ["."]
        
        if self.glob_args:
            dirs = sorted(glob("*" + self.glob_args[0] + "*"),
                          key=lambda x: natural_sorting(x, self.glob_args))
        
        with open(self.outfile, "w") as f:          
            for n, dir in enumerate(dirs):
                if not n:
                    f.write("%s\n" % self._get_header(dir))
                f.write("%s\n" % self._get_data(dir))               


def get_S_matrix_difference(a, b):
    """Print differences between input files.
        
        Parameters:
        -----------
            a, b: str
                Relative paths of S-matrix input files.
    """
    
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
      

def parse_arguments():
    """Parse command-line arguments and call write_S_matrix."""
    
    parser = argparse.ArgumentParser(formatter_class=default_help)

    parser.add_argument("-i", "--infile", default="Smat.sine_boundary.dat",
                        type=str, help="Input file to read S-matrix from.")
    parser.add_argument("-p", "--probabilities", action="store_true",
                        help="Wheter to calculate abs(S)^2.")

    parser.add_argument("-g", "--glob-args", default=[], nargs="*",
                        help="Directory parsing delimiter")
    parser.add_argument("-d", "--delimiter", default="_",
                        type=str, help="Directory parsing delimiter")
    parser.add_argument("-o", "--outfile", default="S_matrix.dat",
                        type=str, help="S-matrix output file.")                        

    parser.add_argument("-D", "--diff", default=[], nargs="*",
                        help="Print difference between input files.")                        
    
    parse_args = parser.parse_args()
    args = vars(parse_args)

    if parse_args.diff:
        get_S_matrix_difference(*parse_args.diff)
    else:
        del args['diff']
        Write_S_Matrix(**args)

   
if __name__ == '__main__':
    parse_arguments()
