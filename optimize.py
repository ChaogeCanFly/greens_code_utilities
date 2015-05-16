#!/usr/bin/env python

from __future__ import division

import glob
import numpy as np
import os
import scipy.integrate
import scipy.optimize
import shutil
import subprocess

import argh

import ep.potential
from helper_functions import replace_in_file


def run_single_job(x, N=None, L=None, W=None, pphw=None, linearized=None,
                   xml_template=None, xml=None, loop_type=None, ncores=None):
    """Prepare and simulate a waveguide with profile

        xi = xi(eps0, delta0, phase0)

    with a parametrization function determined by loop_type.
    """
    eps, delta, phase = x

    ep.potential.write_potential(N=N, pphw=pphw, L=L, W=W, x_R0=eps, y_R0=delta,
                                 init_phase=phase, loop_type=loop_type,
                                 boundary_only=True, shape='RAP', verbose=False,
                                 linearized=linearized)
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
        np.savetxt(f, (eps, delta, phase), newline=" ", fmt='%.16f')
        np.savetxt(f, S, newline=" ", fmt='%.8e')
        f.write("\n")

    return 1. - T01


def run_length_dependent_jobs(x, *single_job_args):
    """Prepare and simulate a waveguide with profile

        xi = xi(eps0, delta0, phase0)

    with a parametrization function determined by loop_type and as a function
    of length L."""

    cwd = os.getcwd()
    args = list(single_job_args)

    L = np.arange(1, args[1], 1)

    for l0 in L:
        args[1] = l0
        ldir = "_L_" + str(l0)
        ldir = os.path.join(cwd, ldir)
        os.mkdir(ldir)
        os.chdir(ldir)
        run_single_job(x, *args)
        os.chdir(cwd)

    cmd = "S_Matrix.py -p -g L -d _L_*"
    subprocess.call(cmd, shell=True)

    # use here area under T01 and T10 as function of length
    # varying parameters stay the same: eps0, delta0, phase0
    L0, T01, T10 = np.loadtxt("S_matrix.dat", unpack=True, usecols=(0, 6, 7))
    A01, A10 = [scipy.integrate.simps(T, L0) for T in (T01, T10)]
    A = (A01 + A10)/2.

    # complementary area fillingfactor
    FF = (1. - A/L)

    # clean directory
    for ldir in glob.glob("_L_*"):
        shutil.rmtree(ldir)

    # archive S_matrices
    num_smatrices = len(glob.glob("S_matrix*dat"))
    shutil.move("S_matrix.dat", "S_matrix_" + str(num_smatrices) + ".dat")
    print "finished iteration #", num_smatrices

    with open("optimize.log", "a") as f:
        data = np.concatenate((x, [A01, A10, A, FF]))
        np.savetxt(f, data, newline=" ", fmt='%.8e')
        f.write("\n")

    return FF


@argh.arg("--xml-template", type=str)
def optimize(eps0=0.2, delta0=0.4, phase0=-1.0, L=10., W=1.,
             length_dependent=False, N=2.5, pphw=100, xml='input.xml',
             xml_template=None, linearized=False, loop_type='Allen-Eberly',
             method='L-BFGS-B', ncores=4, min_tol=1e-5, min_stepsize=1e-2,
             min_maxiter=50):
    """Optimize the waveguide configuration with scipy.optimize.minimize."""

    args = (N, L, W, pphw, linearized, xml_template, xml, loop_type, ncores)
    x0 = (eps0, delta0, phase0)
    bounds = ((0.0, 0.35), (0.0, 3.0), (-5.0, 1.0))
    min_kwargs = {'disp': True,
                  'ftol': min_tol,
                  'maxiter': min_maxiter,
                  'eps': min_stepsize}

    if length_dependent:
        opt_func = run_length_dependent_jobs
    else:
        opt_func = run_single_job

    res = scipy.optimize.minimize(opt_func, x0, args=args, bounds=bounds,
                                  method=method, options=min_kwargs)
    np.save("minimize_res.npy", res)


if __name__ == '__main__':
    argh.dispatch_command(optimize)
