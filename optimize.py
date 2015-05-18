#!/usr/bin/env python

from __future__ import division

import glob
import multiprocessing
import numpy as np
import os
import scipy.integrate
import scipy.optimize
import shutil
import subprocess

import argh

import ep.potential
from helper_functions import replace_in_file


def prepare_and_run_calc(x, N=None, L=None, W=None, pphw=None, linearized=None,
                         xml_template=None, xml=None, loop_type=None,
                         ncores=None):
    """Prepare and simulate a waveguide with profile

        xi = xi(eps0, delta0, phase0)

    with a parametrization function determined by loop_type.
    """
    eps, delta, phase = x

    ep.potential.write_potential(N=N, pphw=pphw, L=L, W=W,
                                 x_R0=eps, y_R0=delta, init_phase=phase,
                                 loop_type=loop_type, boundary_only=True,
                                 shape='RAP', verbose=False,
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

    cmd = "mpirun -np {0} solve_xml_mumps_dev > tmp.out 2>&1".format(ncores)
    subprocess.call(cmd, shell=True)


def run_single_job(x, *job_args):
    """Prepare and simulate a waveguide with profile

        xi = xi(eps0, delta0, phase0)

    with a parametrization function determined by loop_type.
    """
    prepare_and_run_calc(x, *job_args)

    subprocess.call("S_Matrix.py -p", shell=True)
    S = np.loadtxt("S_matrix.dat", unpack=True, usecols=(5, 6))
    T = (S[5] + S[6])/2.

    with open("optimize.log", "a") as f:
        np.savetxt(f, x, newline=" ", fmt='%+15.8f')
        np.savetxt(f, S, newline=" ", fmt='%+.8e')
        f.write("\n")

    return 1. - T


def prepare_dir(x, lengths, cwd, args):
    """Dummy function to allow pickling during multiprocess call."""
    for Ln in lengths:
        args[1] = Ln
        ldir = "_L_" + str(Ln)
        ldir = os.path.join(cwd, ldir)
        os.mkdir(ldir)
        os.chdir(ldir)
        prepare_and_run_calc(x, *args)
        os.chdir(cwd)


def run_length_dependent_job(x, *job_args):
    """Prepare and simulate a waveguide with profile

        xi = xi(eps0, delta0, phase0)

    with a parametrization function determined by loop_type and as a function
    of length L."""

    cwd = os.getcwd()
    args = list(job_args)

    L_list = np.arange(*job_args[1])
    L_parallel = [L_list[n::4] for n in range(4)]

    pool = multiprocessing.Pool(processes=4)
    results = [pool.apply_async(prepare_dir, args=(x, L_batch, cwd, args)) for L_batch in L_parallel]
    results = [p.get() for p in results]

    cmd = "S_Matrix.py -p -g L -d _L_*"
    subprocess.call(cmd, shell=True)

    # use here area under T01 and T10 as function of length
    # varying parameters stay the same: eps0, delta0, phase0
    L, T01, T10 = np.loadtxt("S_matrix.dat", unpack=True, usecols=(0, 6, 7))
    A01, A10 = [scipy.integrate.simps(T, L) for T in (T01, T10)]
    A = (A01 + A10)/2.

    # area fillingfactor; minimize complementary fillingfactor 1 - FF
    FF = A/max(L)

    # clean directory
    for ldir in glob.glob("_L_*"):
        shutil.rmtree(ldir)

    # archive S_matrices
    num_smatrices = len(glob.glob("S_matrix*dat"))
    shutil.move("S_matrix.dat", "S_matrix_" + str(num_smatrices) + ".dat")
    print "finished iteration #", num_smatrices
    with open("optimize.log", "a") as f:
        data = np.concatenate(([num_smatrices], x, [FF]))
        np.savetxt(f, data, newline=" ", fmt='%+.8e')
        f.write("\n")

    return 1. - FF


@argh.arg("--xml-template", type=str)
@argh.arg("-L", "--L", type=float, nargs="+")
def optimize(eps0=0.2, delta0=0.4, phase0=-1.0, L=10., W=1.,
             N=2.5, pphw=100, xml='input.xml', xml_template=None,
             linearized=False, loop_type='Allen-Eberly',
             method='L-BFGS-B', ncores=4, min_tol=1e-5, min_stepsize=1e-2,
             min_maxiter=100):
    """Optimize the waveguide configuration with scipy.optimize.minimize."""

    if len(L) > 1:
        header = "#{:>14}" + 4*" {:>15}"
        header = header.format("iteration", "eps0", "delta0", "phase0", "FF")
        opt_func = run_length_dependent_job
    else:
        header = "#{:>14}" + 10*" {:>15}"
        header = header.format("eps0", "delta0", "phase0",
                               "r00", "r01", "r10", "r11",
                               "t00", "t01", "t10", "t11")
        opt_func = run_single_job
        L = L[0]

    with open("optimize.log", "a") as f:
        f.write(header)
        f.write("\n")

    args = (N, L, W, pphw, linearized, xml_template, xml, loop_type, ncores)
    x0 = (eps0, delta0, phase0)
    bounds = ((0.0, 0.35), (0.0, 3.0), (-5.0, 1.0))
    min_kwargs = {'disp': True,
                  'ftol': min_tol,
                  'maxiter': min_maxiter,
                  'eps': min_stepsize}

    res = scipy.optimize.minimize(opt_func, x0, args=args, bounds=bounds,
                                  method=method, options=min_kwargs)
    np.save("minimize_res.npy", res)


if __name__ == '__main__':
    argh.dispatch_command(optimize)
