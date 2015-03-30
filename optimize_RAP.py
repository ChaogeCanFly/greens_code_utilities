#!/usr/bin/env python

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.optimize
import subprocess
import traceback

import argh

import ep.potential
from helper_functions import replace_in_file


def run_single_job(x, N=None, L=None, W=None, pphw=None,
                   xml_template=None, xml=None, loop_type=None, ncores=None):
    """."""
    eps, delta, phase = x
    ID = "_{loop_type}_L_{L}_eps_{eps:.3f}_delta_{delta:.3f}_phase_{phase:.3f}".format(loop_type=loop_type,
                                                                                       N=N, L=L, eps=eps,
                                                                                       delta=delta,
                                                                                       phase=phase)
    ep.potential.write_potential(N=N, pphw=pphw, L=L, W=W, x_R0=eps, y_R0=delta,
                                 init_phase=phase, loop_type=loop_type,
                                 boundary_only=True, shape='RAP', verbose=False)
    N_file = int(L*(pphw*N+1))
    replacements = {'L"> L':                             'L"> {}'.format(L),
                    'modes"> modes':                     'modes"> {}'.format(N),
                    'wave"> pphw':                       'wave"> {}'.format(pphw),
                    'Gamma0"> Gamma0':                   'Gamma0"> 0.0',
                    'neumann"> neumann':                 'neumann"> 0',
                    'N_file_boundary"> N_file_boundary': 'N_file_boundary"> {}'.format(N_file),
                    'boundary_upper"> boundary_upper':   'boundary_upper"> upper.boundary',
                    'boundary_lower"> boundary_lower':   'boundary_lower"> lower.boundary',
                    }
    replace_in_file(xml_template, xml, **replacements)

    cmd = "mpirun -np {0} solve_xml_mumps_dev > tmp.out 2>&1 && S_Matrix.py -p".format(ncores)
    subprocess.call(cmd, shell=True)

    try:
        S = np.loadtxt("S_matrix.dat")
        with open("optimize.log", "a") as f:
            np.savetxt(f, (eps, delta, phase), newline=" ", fmt='%.6f')
            np.savetxt(f, S, newline=" ", fmt='%.8e')
            f.write("\n")
        T01 = S[5]
        return 1. - T01

    except:
        print traceback.print_exc()


@argh.arg("--L", "--length", type=float)
@argh.arg("--W", "--width", type=float)
@argh.arg("--N", type=float)
@argh.arg("--pphw", type=int)
@argh.arg("--loop-type", type=str)
@argh.arg("--xml-template", type=str)
def optimize_RAP(eps0=0.209, delta0=0.41, phase0=-1.26, L=10., W=1., N=2.5,
                 pphw=100, xml_template=None, xml="input.xml",
                 loop_type='Allen-Eberly-Rubbmark', ncores=4):
    """docstring for optimize_RAP"""
    print "optimize_vars", vars()

    args = (N, L, W, pphw, xml_template, xml, loop_type, ncores)
    x0 = (eps0, delta0, phase0)
    bounds = ((0.1,0.25), (0.1,0.7), (-3.0,0.0))

    res = scipy.optimize.minimize(run_single_job, x0, args=args, bounds=bounds,
                                  method='L-BFGS-B', #callback=write_step,
                                  options={'disp': True, 'maxiter': 50, 'eps': 1e-2})
    print res
    np.save("res.npy", res)


if __name__ == '__main__':
    argh.dispatch_command(optimize_RAP)
