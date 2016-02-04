#!/usr/bin/env python2.7

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import argh

from ep.waveguide import DirichletReduced, DirichletPositionDependentLossReduced


def get_loop_eigenfunction(N=2.6, eta=0.0, L=5., W=1., eps=0.16, nx=10,
                           loop_direction="+", loop_type='Bell', init_state='a',
                           init_phase=0.0, neumann=0):
    """Return the instantaneous eigenfunctions and eigenvectors for each step
    in a parameter space loop.

        Parameters:
        -----------
            N: float
                Number of open modes (floor(N)).
            eta: float
                Dissipation strength.
            L: float
                System length.
            eps: float
                Half-maximum boundary roughness.
            loop_direction: str
                Loop direction of the parameter space trajectory. Allowed
                values: + or -.
            loop_type: str
                Trajectory shape in parameter space.
            init_state: str
                Initial state of the evolution in the effective 2x2 system.
            init_phase: float
                Initial phase in the trajectory.
            neumann: int
                Whether to use Neumann or Dirichlet boundary conditions.
    """

    wg_kwargs = {'N': N,
                 'eta': eta,
                 'L': L,
                 'init_phase': init_phase,
                 'init_state': init_state,
                 'loop_direction': loop_direction,
                 'loop_type': loop_type,
                 'x_R0': eps
    }
    if neumann:
        WG = Neumann(**wg_kwargs)
    else:
        WG = DirichletReduced(**wg_kwargs)
    _, b0, b1 = WG.solve_ODE()

    x = np.linspace(0, L, nx)
    eps, delta = WG.get_cycle_parameters(x)

    for n, (xn, epsn, deltan) in enumerate(zip(x, eps, delta)):
        wgn_kwargs = {'N': N,
                    'eta': eta,
                    'L': 2*np.pi/(WG.kr + deltan),
                    'init_phase': init_phase,
                    'init_state': init_state,
                    'loop_direction': loop_direction,
                    'loop_type': 'Constant',
                    'x_R0': epsn,
                    'y_R0': deltan}
        if neumann:
            WGn = Neumann(**wg_kwargs)
        else:
            WGn = DirichletReduced(**wg_kwargs)
        _, b0, b1 = WGn.solve_ODE()

        Chi_0_eff, Chi_1_eff = WGn.eVecs_r[:,:,0], WGn.eVecs_r[:,:,1]

        # switch eigenvectors halfway
        if n > len(x)/2:
            Chi_0_eff, Chi_1_eff = Chi_1_eff, Chi_0_eff

        K_0_eff, K_1_eff = WGn.eVals[:,0], WGn.eVals[:,1]

        xnn = WGn.t
        yN = len(xnn)/WGn.T
        y = np.linspace(0, WGn.W, yN)
        X, Y = np.meshgrid(xnn, y)

        # assemble effective model eigenvectors
        Chi_0_eff[:,0] *= np.exp(1j*K_0_eff*xnn)
        Chi_0_eff[:,1] *= np.exp(1j*K_0_eff*xnn)
        Chi_1_eff[:,0] *= np.exp(1j*K_1_eff*xnn)
        Chi_1_eff[:,1] *= np.exp(1j*K_1_eff*xnn)

        if neumann:
            Chi_0_eff_0 = np.outer(Chi_0_eff[:,0], 1.*np.ones_like(y))
            Chi_0_eff_1 = np.outer(Chi_0_eff[:,1]*np.exp(-1j*WGn.kr*xnn),
                                np.sqrt(2.*WGn.k0/WGn.k1)*np.cos(np.pi*y))
            Chi_0_eff = Chi_0_eff_0 + Chi_0_eff_1

            Chi_1_eff_0 = np.outer(Chi_1_eff[:,0], 1.*np.ones_like(y))
            Chi_1_eff_1 = np.outer(Chi_1_eff[:,1]*np.exp(-1j*WGn.kr*xnn),
                                np.sqrt(2.*WGn.k0/WGn.k1)*np.cos(np.pi*y))
            Chi_1_eff = Chi_1_eff_0 + Chi_1_eff_1
        else:
            Chi_0_eff_0 = np.outer(Chi_0_eff[:,0], np.sin(np.pi*y))
            Chi_0_eff_1 = np.outer(Chi_0_eff[:,1]*np.exp(-1j*WGn.kr*xnn),
                                np.sqrt(WGn.k1/WGn.k0)*np.sin(2*np.pi*y))
            Chi_0_eff = Chi_0_eff_0 + Chi_0_eff_1

            Chi_1_eff_0 = np.outer(Chi_1_eff[:,0], np.sin(np.pi*y))
            Chi_1_eff_1 = np.outer(Chi_1_eff[:,1]*np.exp(-1j*WGn.kr*xnn),
                                np.sqrt(WGn.k1/WGn.k0)*np.sin(2*np.pi*y))
            Chi_1_eff = Chi_1_eff_0 + Chi_1_eff_1

        Chi_0_eff, Chi_1_eff = [ c.T for c in Chi_0_eff, Chi_1_eff ]

        ID = "n_{:03}_xn_{:08.4f}_epsn_{}_deltan_{}".format(n, xn, epsn, deltan)
        print ID, WGn.kr
        part = np.abs
        f, (ax1, ax2) = plt.subplots(nrows=2)

        ax1.pcolormesh(X, Y, part(Chi_0_eff), vmin=np.abs(Chi_0_eff).min(), vmax=np.abs(Chi_0_eff).max())
        ax1.contour(X, Y, np.real(Chi_0_eff), levels=[0.0], colors='k', linestyles="solid")
        ax1.contour(X, Y, np.imag(Chi_0_eff), levels=[0.0], colors='w', linestyles="dashed")

        ax2.pcolormesh(X, Y, part(Chi_1_eff), vmin=np.abs(Chi_1_eff).min(), vmax=np.abs(Chi_1_eff).max())
        ax2.contour(X, Y, np.real(Chi_1_eff), levels=[0.0], colors='k', linestyles="solid")
        ax2.contour(X, Y, np.imag(Chi_1_eff), levels=[0.0], colors='w', linestyles="dashed")

        plt.savefig(ID + ".png")


if __name__ == '__main__':
    argh.dispatch_command(get_loop_eigenfunction)
