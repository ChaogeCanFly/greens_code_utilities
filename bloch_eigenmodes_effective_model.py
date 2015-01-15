#!/usr/bin/env python2.7

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import numpy as np

import argh

from ep.waveguide import Waveguide


def get_loop_eigenfunction(N=1.05, eta=0.0, L=5., d=1., eps=0.05, nx=None,
                           loop_direction="+", loop_type='Bell', init_state='a',
                           init_phase=0.0, mpi=False, pphw=100,
                           effective_model_only=False, neumann=1):
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
                 'neumann': neumann}
    WG = Waveguide(**wg_kwargs)
    WG.x_EP = eps
    _, b0, b1 = WG.solve_ODE()

    x = np.linspace(0, L, 10)
    eps, delta = WG.get_cycle_parameters(x)

    for n, (xn, epsn, deltan) in enumerate(zip(x, eps, delta)):
        wg_kwargs = {'N': N,
                    'eta': eta,
                    'L': L,
                    'init_phase': init_phase,
                    'init_state': init_state,
                    'loop_direction': loop_direction,
                    'loop_type': 'Constant',
                    'neumann': neumann}
        WGn = Waveguide(**wgn_kwargs)
        WGn.x_EP = epsn
        WGn.y_EP = deltan
        ID = "n_{:03}_xn_{:08.4f}_epsn_{}_deltan_{}".format(n, xn, epsn, deltan)
        _, b0, b1 = WGn.solve_ODE(instantaneous_eigenbasis=True,
                                  save_plot=ID + ".png")

        WGn.draw_wavefunction()


if __name__ == '__main__':
    argh.dispatch_command(get_loop_eigenfunction)
