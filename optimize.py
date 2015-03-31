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
    """Prepare and simulate a waveguide with profile

        xi = xi(eps0, delta0, phase0)

    with a parametrization function determined by loop_type.
    """
    eps, delta, phase = x

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
    S = np.loadtxt("S_matrix.dat")
    T01 = S[5]

    with open("optimize.log", "a") as f:
        np.savetxt(f, (eps, delta, phase), newline=" ", fmt='%.8f')
        np.savetxt(f, S, newline=" ", fmt='%.8e')
        f.write("\n")

    return 1. - T01


@argh.arg("--N0", type=float)
@argh.arg("--xml-template", type=str)
def optimize(eps0=0.2, delta0=0.4, phase0=-1.0, N0=None, L=10., W=1.,
             N=2.5, pphw=100, xml="input.xml", xml_template=None,
             loop_type='Allen-Eberly-Rubbmark', method='L-BFGS-B',
             ncores=4, min_tol=1e-5, min_stepsize=1e-2, min_maxiter=50):
    """Optimize the waveguide configuration with scipy.optimize.minimize."""

    args = (N, L, W, pphw, xml_template, xml, loop_type, ncores)
    x0 = (eps0, delta0, phase0)
    bounds = ((0.0,0.5), (0.0,1.0), (-5.0,5.0))
    min_kwargs = {'disp': True,
                  'ftol': min_tol,
                  'maxiter': min_maxiter,
                  'eps': min_stepsize}

    res = scipy.optimize.minimize(run_single_job, x0, args=args, bounds=bounds,
                                  method=method, options=min_kwargs)
    np.save("minimize_res.npy", res)


if __name__ == '__main__':
    argh.dispatch_command(optimize)
