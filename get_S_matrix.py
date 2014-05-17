#!/usr/bin/env python2.7

import numpy as np
import argparse
from glob import glob
import os

class S_Matrix:
    """S-matrix class"""

    def __init__(self, indir=".", infile="Smat.sine_boundary.dat",
                 outfile="S_matrix.dat", probabilities=False, d="-",
                 method="amplitudes"):
        """Reads and processes the S-matrix.
            
            Parameters:
            -----------
                indir: str
                    Input directory.
                infile: str
                    Input file to read S-matrix from.
                outfile: str
                    S-matrix output file.
                probabilities: bool
                    Whether to calculate the absulte square abs(S)^2.
                d: str ('-'|'+')
                    Loop direction.
                method: str
                    Processing method.
        """

        self.indir = indir
        self.infile = indir + "/" + infile
        self.outfile = outfile
        self.probabilities = probabilities
        self.d = d
        #self.custom_glob = custom_glob
        try:
            self._get_amplitudes()
        except IndexError as e:
            #print "{}: missing values in directory {}.".format(e,directory)
            self.S = np.zeros((4,4))
            
    def __str__(self):
        """Print reflection/transmission amplitudes/probabilities."""
        return self._get_data()
   
    def _get_amplitudes(self):
        """Get transmission and reflection amplitudes."""
        
        # get number of scheduler steps and S-matrix dimensions
        nruns, ndims = np.loadtxt(self.infile)[:3:2]

        # get real and imaginary parts of S-matrix
        re, im = np.genfromtxt(self.infile, usecols=(2,3), autostrip=True,
                               unpack=True, invalid_raise=False)

        # calculate transmission and reflection amplitudes
        S = (re + 1j*im).reshape((nruns,ndims,ndims))
        print S
        if self.probabilities:
            self.S = S
        else:
            self.S = abs(S)**2
        self.nruns = int(nruns)
        self.ndims = int(ndims)

    def get_header(self):
        """Prepare data file header."""
        
        # tune alignment spacing
        spacing = 17 if self.probabilities else 35
        
        header = [ "{}{}{}".format(s,i,j) for s in ("t","r") 
                                          for i in range(self.ndims//2) 
                                          for j in range(self.ndims//2) ]   
        headerfmt = '# {:>10}  {:>12}        '          
        headerfmt += "  ".join([ '{:>{s}}' for n in range(self.ndims*self.ndims//2) ])
        
        return headerfmt.format('eta', 'L', *header, s=spacing)       
        
    def _get_data(self, method='amplitudes'):
        """Prepare S-matrix data for output."""
            
        data = [ self.S[i,j,k] for i in range(self.nruns) 
                               for j in range(self.ndims)
                               for k in range(self.ndims//2)]       
        datafmt = '  {:>10}  {:>12}        '          
        datafmt += "  ".join([ '{:> .10e}' for n in range(self.ndims*self.ndims//2) ])

        if method == 'heatmap':
            eta, L = [ self.indir.split("_")[i] for i in (1,3) ]
        elif method == 'length':
            eta, L = [ self.indir.split("_")[i] for i in (1,5) ]
        elif method == 'amplitudes':
            eta, L = 0., 0.

        return datafmt.format(eta, L, *data)

    def write(self, method='length'):
        """Write S_matrix to outfile."""
    
        if method == 'amplitude':
            with file(self.outfile, 'w') as out:
                out.write('# S-matrix shape: {0}\n'.format(self.S.shape))
                for i, Si in enumerate(self.S):
                    out.write('# loss iteration {0}\n'.format(i))
                    np.savetxt(self.outfile, Si)

        elif method == 'length' or method == 'heatmap':
            
            # tune alignment spacing
            spacing = 17 if self.probabilities else 35
            
            with file(self.outfile, 'w') as f:  
                
                header = self._get_header()
                f.write(header)

                if self.custom_glob:
                    folders = sorted(glob(custom_glob),
                                     key=lambda x: float(x.split("_")[-1]))
                else:
                    folders = sorted(glob("eta*_L*_{0}".format(d)),
                                     key=lambda x: float(x.split("_")[-2]))
            
                for directory in folders:
                    print "Processing directory ", directory

                    if method == 'heatmap':
                        eta, L = [ directory.split("_")[i] for i in (1,3) ]
                    elif method == 'length':
                        eta, L = [ directory.split("_")[i] for i in (1,5) ]
            
#                     S_sliced = [ self.S[i,j] for i in range(self.ndims)
#                                              for j in range(self.ndims) ]
                    f.write()  

                    
class Multiple_S_Matrices():
    """
        Parameters:
        -----------
            custom_glob: list
                List of non-standard input directories.
    """
    def __init__(self, custom_glob=None):
        pass
        
    def _parse_directory(self):
        """Extract variables from directory names."""
        # get correct ordering
        if self.custom_glob:
            folders = sorted(glob(self.custom_glob),
                             key=lambda x: float(x.split("_")[-1]))
        else:
            folders = sorted(glob("eta*_L*_{0}".format(d)),
                             key=lambda x: float(x.split("_")[-2]))    
        return folders
    
    pass             
                    
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

    parser = argparse.ArgumentParser()
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
