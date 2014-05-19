#!/usr/bin/env python2.7

import re
import numpy as np
import argparse
from glob import glob
import os

class S_Matrix:
    """S-matrix class"""

    def __init__(self, indir=".", 
                 infile="Smat.sine_boundary.dat",
                 probabilities=False,
                 glob_args=[], delimiter="_"):
        """Reads and processes the S-matrix.
            
            Parameters:
            -----------
                indir: str
                    Input directory.
                infile: str
                    Input file to read S-matrix from.
                probabilities: bool
                    Whether to calculate abs(S)^2.
                glob_args: list of strings
                    Directory parsing parameters.
                delimiter: str
                    Directory parsing delimiter.
        """

        self.indir = indir
        self.infile = indir + "/" + infile
        self.glob_args = glob_args
        self.nargs = len(glob_args) if glob_args else 0
        self.probabilities = probabilities
        
        self.delimiter = delimiter
        #self.glob_pattern = "*".join([ g + d for g in glob_args 
        #                                     for d in delimiter ]) + "*"
        #self._get_amplitudes()

    def __str__(self):
        """Print reflection/transmission amplitudes/probabilities."""
        return self.data 

    def _get_amplitudes(self):
        """Get transmission and reflection amplitudes."""
        
        # get number of scheduler steps and S-matrix dimensions
        nruns, ndims = np.loadtxt(self.infile)[:3:2]

        try:
            # get real and imaginary parts of S-matrix
            re, im = np.genfromtxt(self.infile, usecols=(2,3), autostrip=True,
                                   unpack=True, invalid_raise=False)
            # calculate transmission and reflection amplitudes
            S = (re + 1j*im).reshape((nruns,ndims,ndims))
        except:
            # initialize zero_like S-matrix if no data available
            S = np.zeros((self.ndims,self.ndims))
            
        if not self.probabilities:
            self.S = S
        else:
            self.S = abs(S)**2
            
        self.nruns = int(nruns)
        self.ndims = int(ndims)    
        
    def _parse_directory(self):
        """Extract running variables from directory name."""

        dir = self.indir.split(self.delimiter)
        arg_values = []

        try:
            for p in self.glob_args:
                idx = dir.index(p)
                arg_values.append(dir[idx+1])
        except ValueError:
            arg_values.append([])
            
        self.arg_values = arg_values

    def get_header(self):
        """Prepare data file header."""
        
        self._get_amplitudes()
        
        # tune alignment spacing
        spacing = 17 if self.probabilities else 35
        
        # translate S-matrix into transmission and reflection components
        header = [ "{}{}{}".format(s,i,j) for s in ("r","t") 
                                          for i in range(self.ndims//2) 
                                          for j in range(self.ndims//2) ]
        headerdim = self.ndims*self.ndims//2  
        
        headerfmt = '#'
        headerfmt += "  ".join([ '{:>12}' for n in range(self.nargs) ]) + "  "     
        headerfmt += "  ".join([ '{:>{s}}' for n in range(headerdim) ])
        
        self.header = headerfmt.format(*(self.glob_args + header), s=spacing) 
        
        return self.header      
        
    def get_data(self):
        """Prepare S-matrix data for output."""
            
        self._parse_directory()    
        self._get_amplitudes()
        
        data = [ self.S[i,j,k] for i in range(self.nruns) 
                               for j in range(self.ndims)
                               for k in range(self.ndims//2) ]
        datadim = self.ndims*self.ndims//2   
     
        datafmt = " "
        datafmt += "  ".join([ '{:>12}' for n in range(self.nargs) ]) + "  "
        datafmt += "  ".join([ '{:> .10e}' for n in range(datadim) ])
        
        self.data = datafmt.format(*(self.arg_values + data))
        
        return self.data


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
    


def write_S_matrix(outfile="S_matrix.dat", 
                   **kwargs):
    """Write S-matrix data to file.
    
        Parameters:
        -----------
            outfile: str
                S-matrix output file.
    
        For kwargs, see S_Matrix class.
    """

    S = S_Matrix(**kwargs)
    glob_args = kwargs['glob_args']

    if glob_args:
        dir_list = sorted(glob("*" + glob_args[0] + "*"),
                          key=lambda x: natural_sorting(x, glob_args))
    else:
        dir_list = ["."]
    
    with open(outfile, "w") as f:    
        kwargs['indir'] = dir_list[0]        
        S = S_Matrix(**kwargs)
        f.write("%s\n" % S.get_header())
        for dir in dir_list:
            kwargs['indir'] = dir        
            S = S_Matrix(**kwargs)
            f.write("%s\n" % S.get_data())   
      
      
def parse_arguments():
    """Parse command-line arguments and call write_S_matrix."""
    
    parser = argparse.ArgumentParser()

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
    
    parse_args = vars(parser.parse_args())

    #print parse_args
    write_S_matrix(**parse_args)

   
if __name__ == '__main__':
    parse_arguments()
