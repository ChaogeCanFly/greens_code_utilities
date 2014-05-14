#!/usr/bin/env python2.7

import numpy as np
import argparse as ap
from glob import glob
import os

class S_Matrix:
    """S-matrix class"""
    def __init__(self, indir=None, outfile=None, probabilities=False, d='-'):
        self.input_directory = os.getcwd() if indir else indir
        self.outfile = outfile              # S-matrix output file
        self.probabilities = probabilities  # return abs(S)**2
        self.d = d                          # loop direction
        self.get_amplitudes()               # initialize S, nloss and ndims
    
    def get_amplitudes(self):
        """Calculate transmission and reflection amplitudes."""

        # get number of dissipation-steps and S-matrix dimensions
        infile = glob("{0}/Smat.*.dat".format(self.input_directory))[0]
        nloss, ndims = np.loadtxt(infile)[:2]

        # get real and imaginary parts of S-matrix
        re, im = np.genfromtxt(infile, usecols=(2,3), autostrip=True,
                               unpack=True, invalid_raise=False)

        # calculate transmission and reflection amplitudes
        S = (re + 1j*im).reshape((nloss,ndims,ndims))
    
        self.S = S
        self.nloss = nloss
        self.ndims = ndims
        
    def write(self, method=None):
        """Write S_matrix to outfile."""
    
        if method == 'amplitude':
            with file(self.outfile, 'w') as out:
                S = get_amplitudes("{0}/{1}".format(cwd,directory), 
                                   probabilities=probabilities)[0]
                out.write('# S-matrix shape: {0}\n'.format(S.shape))
                for i, Si in enumerate(S):
                    out.write('# loss iteration {0}\n'.format(i))
                    np.savetxt(outfile, Si)
        elif method == 'length' or method == 'heatmap':
            # tune alignment spacing
            spacing = 17 if probabilities else 35
            
            with file(self.outfile, 'w') as f:  
                f.write(('# {:>10}  {:>12}        '
                         '{:>{s}}  {:>{s}}  {:>{s}}  '
                         '{:>{s}}  {:>{s}}  {:>{s}}  '
                         '{:>{s}}  {:>{s}}  \n').format("eta", "L",
                                                        "t00", "t01", "t10", "t11",
                                                        "r00", "r01", "r10", "r11",
                                                        s=spacing))
                # get correct ordering
                folders = sorted(glob("eta*_L*_{0}".format(d)), 
                                 key=lambda x: float(x.split("_")[-2]))
            
                for directory in folders:
                    print "Processing directory ", directory

                    if method == 'heatmap':
                        eta, L = [ directory.split("_")[i] for i in (1,3) ]
                    elif method == 'length':
                        eta, L = [ directory.split("_")[i] for i in (1,5) ]
            
                    S = get_amplitudes("{0}/{1}".format(cwd,directory), 
                                        probabilities=probabilities)[0]
                    
                    # reflection amplitudes
                    r00 = S[0,0]
                    r01 = S[0,1]
                    r10 = S[1,0]
                    r11 = S[1,1]
                
                    # transmission amplitudes       
                    t00 = S[2,0]
                    t01 = S[2,1]
                    t10 = S[3,0]
                    t11 = S[3,1]
                
                    f.write(('{:>12}  {:>12}        '
                             '{:> .10e}  {:> .10e}  '
                             '{:> .10e}  {:> .10e}  '
                             '{:> .10e}  {:> .10e}  '
                             '{:> .10e}  {:> .10e}\n').format(eta, L, 
                                                              t00, t01, t10, t11,
                                                              r00, r01, r10, r11))
                    
                    
                    
def get_amplitudes(input_dir=None, probabilities=False, 
           outfile=None, diofile="Diodicities.dat"):
    """Calculate transmission and reflection amplitudes."""

    # get number of dissipation-steps and S-matrix dimensions
    if input_dir is None:
        input_dir="."
    infile = glob("{0}/Smat.*.dat".format(input_dir))[0]
    nloss, ndims = np.loadtxt(infile)[:2]

    if not ndims.is_integer():
        ndims = np.loadtxt(infile)[2]

    # get real and imaginary parts of S-matrix
    re, im = np.genfromtxt(infile, usecols=(2,3), autostrip=True,
                   unpack=True, invalid_raise=False)

    # calculate transmission and reflection amplitudes
    S = (re + 1j*im).reshape((nloss,ndims,ndims))
    if probabilities:
        S = np.abs(S)**2

    if outfile:
        with file(outfile, 'w') as outfile:
            with file(diofile, 'w') as diofile:
                diofile.write(('# {0:>10}  {1:>16}'
                            '  {2:>16}  {3:>16}'
                            '  {4:>16}\n').format("iteration", "t00", 
                                                    "t10", "t01", "t01"))
                outfile.write('# S-matrix shape: {0}\n'.format(S.shape))
        
                for i, Si in enumerate(S):
                    t00 = Si[2,0]
                    t01 = Si[2,1]
                    t10 = Si[3,0]
                    t11 = Si[3,1]
                    diofile.write(('{0:>12}  {1:>.10e}  '
                                '{2:>.10e}  {3:>.10e}  '
                                '{4:>.10e}\n').format(i, t00, t10, t01, t11))
                    outfile.write('# loss iteration {0}\n'.format(i))
                    np.savetxt(outfile, Si)
    else:
        return S


def get_amplitudes_heatmap(outfile=None, d='-'):
    """Scan and read transmission and reflection amplitudes for folder array."""
    
    cwd = os.getcwd()
    
    # tune alignment spacing
    spacing = 17 if probabilities else 35
    
    with file(outfile, 'w') as f:   
        f.write(('# {:>10}  {:>12}        '
                '{:>{s}}  {:>{s}}  {:>{s}}  '
                '{:>{s}}  {:>{s}}  {:>{s}}  '
                '{:>{s}}  {:>{s}}  \n').format("eta", "Ln",
                                               "t00", "t01", "t10", "t11",
                                               "r00", "r01", "r10", "r11",
                                               s=spacing))
        
        for directory in glob("eta*_L*_{0}".format(d)):
            print "Processing directory ", directory
            
            eta, L = [ directory.split("_")[i] for i in (1,3) ]
        
            S = get_amplitudes("{0}/{1}".format(cwd,directory), 
                                probabilities=probabilities)[0]
            
            # reflection amplitudes
            r00 = S[0,0]
            r01 = S[0,1]
            r10 = S[1,0]
            r11 = S[1,1]
            
            # transmission amplitudes       
            t00 = S[2,0]
            t01 = S[2,1]
            t10 = S[3,0]
            t11 = S[3,1]
            
            f.write(('{:>12}  {:>12}        '
                     '{:> .10e}  {:> .10e}  '
                     '{:> .10e}  {:> .10e}  '
                     '{:> .10e}  {:> .10e}  '
                     '{:> .10e}  {:> .10e}\n').format(eta, L, 
                                                       t00, t01, t10, t11,
                                                       r00, r01, r10, r11))


def get_amplitudes_vs_length(outfile=None, d='-', probabilities=False, custom_glob=None):
    """
    Scan and read length dependent transmission amplitudes for folder array.
    """
    
    cwd = os.getcwd()
    
    # tune alignment spacing
    spacing = 17 if probabilities else 35
    
    with file(outfile, 'w') as f:   
        f.write(('# {:>10}  {:>12}        '
                '{:>{s}}  {:>{s}}  {:>{s}}  '
                '{:>{s}}  {:>{s}}  {:>{s}}  '
                '{:>{s}}  {:>{s}}  \n').format("eta", "Ln",
                                               "t00", "t01", "t10", "t11",
                                               "r00", "r01", "r10", "r11",
                                               s=spacing))
        # get correct ordering
        if custom_glob:
            folders = sorted(glob(custom_glob), key=lambda x: float(x.split("_")[-1]))
        else:
            folders = sorted(glob("eta*_L*_{0}".format(d)),
                             key=lambda x: float(x.split("_")[-2]))
        
        for directory in folders:
            print "Processing directory ", directory

            if not custom_glob:
                eta, Ln = [ directory.split("_")[i] for i in (1,5) ]
            else:
                eta, Ln = [ directory.split("_")[i] for i in (1,2) ]
            
            try:
                S = get_amplitudes("{0}/{1}".format(cwd,directory), 
                                   probabilities=probabilities)[0]
                S[2,0]
            except IndexError as e:
                print "{}: missing values in directory {}.".format(e,directory)
                S = np.zeros((4,4))
            
            # reflection amplitudes
            r00 = S[0,0]
            r01 = S[0,1]
            r10 = S[1,0]
            r11 = S[1,1]
            
            # transmission amplitudes       
            t00 = S[2,0]
            t01 = S[2,1]
            t10 = S[3,0]
            t11 = S[3,1]
            
            f.write(('{:>12}  {:>12}        '
                     '{:> .10e}  {:> .10e}  '
                     '{:> .10e}  {:> .10e}  '
                     '{:> .10e}  {:> .10e}  '
                     '{:> .10e}  {:> .10e}\n').format(eta, Ln, 
                                                       t00, t01, t10, t11,
                                                       r00, r01, r10, r11))
     
    
if __name__ == '__main__':

    parser = ap.ArgumentParser()
    parser.add_argument("-o", "--output-file", default="S_matrix.dat",
                        type=str, help="S-matrix output file")
    parser.add_argument("-d", "--direction", default="-",
                        type=str, help="Loop direction")
    parser.add_argument("-p", "--probabilities", action="store_true",
                        help="Return |S|^2")
    parser.add_argument("-c", "--custom-glob", default=None, type=str,
                        help="Set custom input file")
    parser.add_argument("-i", "--input-dir", default=".", type=str,
                        help="Set input directory")
                        
    method = parser.add_mutually_exclusive_group(required=True)
    method.add_argument("-a", "--amplitudes", action="store_true",
                        help="Calculate amplitudes")
    method.add_argument("-m", "--heatmap", action="store_true",
                        help="Calculate amplitudes for heatmap")
    method.add_argument("-l", "--length", action="store_true",
                        help="Calculate length dependent amplitudes")
                        
    args = parser.parse_args()

    outfile = args.output_file
    direction = args.direction
    probabilities = args.probabilities
    custom_glob = args.custom_glob

    if args.amplitudes:
        get_amplitudes(input_dir=args.input_dir, outfile=outfile, 
                   probabilities=probabilities)
    elif args.heatmap:
        get_amplitudes_heatmap(outfile=outfile, d=direction)
    elif args.length:
        get_amplitudes_vs_length(outfile=outfile, d=direction,
                     probabilities=probabilities, custom_glob=custom_glob)
