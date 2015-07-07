#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy.integrate
import scipy.optimize
import subprocess

import argh

from S_Matrix import test_S_matrix_symmetry


def run_single_job(x):

    cmd = "./run.sh {} {} {} {}".format(*x)
    subprocess.check_call(cmd.split())

    S, F = test_S_matrix_symmetry("Smat.complex_potential.dat")

    with open("optimize.log", "a") as f:
        np.savetxt(f, np.concatenate([x, [F]]), newline=" ", fmt='%+3.8f')
        f.write("\n")
        np.savetxt(f, S, fmt='# %+3.5e')
        f.write("\n")

    return F


def optimize(thresh0=1e-2, sx0=6.5, sy0=1.05, ampl0=2e4,
             ncores=4, algorithm='minimize', method='L-BFGS-B',
             min_tol=1e-5, min_stepsize=1e-2, min_maxiter=100):
    """Optimize the waveguide configuration with scipy.optimize.minimize."""

    opt_func = run_single_job

    BOUNDS = ((0.0, 5e-2), (0.0, 30.0), (0.0, 30.0), (0.0, 1e5))

    if algorithm == 'minimize':
        x0 = (thresh0, sx0, sy0, ampl0)
        min_kwargs = {'disp': True,
                      'ftol': min_tol,
                      'maxiter': min_maxiter,
                      'eps': min_stepsize}
        res = scipy.optimize.minimize(opt_func, x0, bounds=BOUNDS,
                                      method=method, options=min_kwargs)

    elif algorithm == 'differential_evolution':
        de_kwargs = {'disp': True,
                     'tol': min_tol,
                     'maxiter': min_maxiter}
        res = scipy.optimize.differential_evolution(opt_func,
                                                    bounds=BOUNDS,
                                                    **de_kwargs)

    np.save("minimize_res.npy", res)


if __name__ == '__main__':
    argh.dispatch_command(optimize)
