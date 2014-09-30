#!/usr/bin/env python2.7

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys

import argh

import bloch
from helpers import replace_in_file
from ep.waveguide import Waveguide
from xmlparser import XML


class Jordan(object):
    """Find the Jordan-block of a multiple eigenvalue problem.

        Parameters:
        -----------
            x0, y0: float
                Initial guess of the parameters.
            dx, dy: float
                Estimated steps to find x1 and y1.
            rtol: float
                Relative tolerance for convergence measure.
            executable: str
                Program name to call during optimization.
            outfile: str
                Parameter output file.
            evalsfile: str
                Eigenvalue file.
            xml: str
                Input .xml file name.
            template: str
                Input .xml template name.
            waveguide_params:
                Parameters of the ep.waveguide.Waveguide class.
            interactive: bool
                Whether to run the program interactively.
            pphw: int
                Number of grid-points per half wavelength.

        Attributes:
        -----------
            values: list of lists
                Accumulates the parameters x, y during optimization.
            evals: list of lists
                Accumulates the Bloch eigenvalues during optimization.
            residual: float
                Residual vector |v_i+1 - v_i|^2.
            dx: float
                Grid discretization.
    """

    def __init__(self, x0, y0, dx=1e-2, dy=1e-2, rtol=1e-6,
                 executable='solve_xml_mumps', outfile='jordan.out',
                 evalsfile='Evals.sine_boundary.dat',
                 template='input.xml_template', pphw=100,
                 xml='input.xml', interactive=False, **waveguide_params):

        self.values = [[x0, y0], [x0+dx, y0+dy]]
        self.evals = []
        self.rtol = rtol
        self.residual = 1.

        self.executable = executable
        self.outfile = outfile
        self.evalsfile = evalsfile
        self.template = template
        self.pphw = pphw
        self.xml = xml
        self.waveguide_params = waveguide_params

        self.interactive = interactive

        self._update_boundary(x0, y0)
        self.__dict__.update(XML(xml).params)

    def _setup(self):
        pass

    def _run_code(self):
        params = {'E': self.executable,
                  'I': self.xml}
        cmd = "subSGE.py -l -e {E} -i {I}".format(**params)
        subprocess.call(cmd.split())

    def _iterate(self):
        (x0, y0), (x1, y1) = self.values[-2:]
        dx, dy = x1-x0, y1-y0
        dx = 1.1 * self.dx
        dy = 1.1 * self.dx

        if abs(dx) < self.dx or abs(dy) < self.dx:
            print """
                WARNING: dx or dy smaller than discretization!

                abs(dx) = {0}
                abs(dy) = {1}
                self.dx = {2}
            """.format(abs(dx), abs(dy), self.dx)
            self._print_and_save()
            sys.exit()
            # dx = 1.1 * self.dx
            # dy = 1.1 * self.dx

        print "x0, x1", x0, x1
        print "y0, y1", y0, y1
        print "dx, dy", dx, dy

        eigenvalues = []
        parameters = [ (n,m) for n in x0, x1 for m in y0, y1 ]
        # order: (x0, y0) -> (x_i,   y_i)
        #        (x0, y1) -> (x_i,   y_i+1)
        #        (x1, y0) -> (x_i+1, y_i)
        #        (x1, y1) -> (x_i+1, y_i+1)
        if 1:
            for n, (x, y) in enumerate(parameters):
                # we dont need the values lambda_n(x_i+1, y_i+1)
                if n < 3:
                    print "n,m", x, y
                    self._check_numerical_resolution(x)
                    self._update_boundary(x, y)
                    self._run_code()
                    # take the first two right-movers & k1 -> k1 mod kr
                    # TODO: why column 0 and not 1 to access the right moving modes?
                    bloch_modes = np.array(bloch.get_eigensystem())[0,:2]
                    F = np.abs(bloch_modes[0] - bloch_modes[1])
                    print F == np.abs(np.array([1,-1]).dot(bloch_modes))
                    eigenvalues.append(F)
        else:
            # alternative
            pass

        self.evals.append([bloch_modes[0],
                           bloch_modes[1]])
        delta = F/2.
        F = np.asarray(eigenvalues).T

        gradient = np.array([[-1, 0, 1],
                             [-1, 1, 0]])

        gradient_x, gradient_y = gradient.dot(F)

        gradient_x /= dx
        gradient_y /= dy

        normsq = (np.abs(gradient_x)**2 +
                  np.abs(gradient_y)**2)

        res_x, res_y  = [ g/normsq*delta for g in gradient_x, gradient_y ]

        print "res_x", res_x
        print "res_y", res_y

        x2 = x1 - res_x
        y2 = y1 - res_y

        self.residual = np.sqrt(np.abs(res_x)**2 + np.abs(res_y)**2)

        return x2, y2

    def _check_numerical_resolution(self, x):
        print """
            Numerical resolution:

                x:      {0}
                dx:     {1}
                x/dx:   {2}
        """.format(x, self.dx, int(x/self.dx))

    def _update_boundary(self, x, y):

        N = self.waveguide_params.get("N")
        eta = self.waveguide_params.get("eta")
        print "N:", N
        print "eta: ", eta

        k0, k1 = [ np.sqrt(N**2 - n**2)*np.pi for n in 0, 1 ]
        L = abs(2*np.pi/(k0 - k1 + y))

        WG = Waveguide(L=L, loop_type='Constant', **self.waveguide_params)
        self.WG = WG

        print "WG.x_EP", WG.x_EP
        print "WG.y_EP", WG.y_EP

        WG.x_EP = x
        WG.y_EP = y

        xi_lower, xi_upper = WG.get_boundary(eps=x, delta=y)

        np.savetxt("lower.profile", zip(WG.t, xi_lower))
        np.savetxt("upper.profile", zip(WG.t, xi_upper))

        N_file = len(WG.t)
        replacements = {'L"> L':           'L"> {}'.format(L),
                        'wave"> pphw':     'wave"> {}'.format(self.pphw),
                        'N_file"> N_file': 'N_file"> {}'.format(N_file),
                        'Gamma0"> Gamma0': 'Gamma0"> {}'.format(eta)}

        replace_in_file(self.template, self.xml, **replacements)

    def _print_and_save(self):
        v1, v2 = np.array(self.values).T
        e1, e2 = np.array(self.evals).T
        for n, a, b, c, d in zip(range(len(v1)), v1, v2, e1, e2):
            print n, a, b, c.real, c.imag, d.real, d.imag, np.abs(c-d)
        np.savetxt(self.outfile, zip(range(len(v1)), v1, v2, e1.real, e1.imag, 
                                 e2.real, e2.imag, np.abs(e1-e2)), fmt="%.6f",
                   header='n eps delta Re(K0) Im(K0) Re(K1) Im(K1) abs(K0-K1)')

    def solve(self):
        # while self.residual > self.rtol:
        for n in range(10):
            xi, yi = self._iterate()
            self.values.append([xi, yi])

            print "residual", self.residual
            print "rtol", self.rtol
            print "x, y:", self.values[-1]

            if self.interactive:
                prompt = raw_input("Continue? (y)es/(n)o/(p)lot ")
            else:
                prompt = "y"

            if not prompt:
                pass
            elif prompt[0] == 'n':
                self._print_and_save()
                sys.exit()
            elif prompt[0] == 'p':
                f = plt.figure(0)
                x, y = np.array(self.values).reshape(-1,2).T
                plt.plot(x, y, "ro-")
                for n, (x, y) in enumerate(self.values):
                    plt.text(x, y, str(n), fontsize=12)
                plt.show()
            else:
                pass

        self._print_and_save()


@argh.arg('x0', type=float)
@argh.arg('y0', type=float)
@argh.arg('-N', '--N', type=float)
@argh.arg('--eta', type=float)
def find_EP(x0, y0, dx=1e-2, dy=1e-2, rtol=1e-6, executable='solve_xml_mumps',
            outfile='jordan.out', evalsfile='Evals.sine_boundary.dat', pphw=100,
            xml='input.xml', template='input.xml_template', interactive=False,
            **waveguide_params):
    J = Jordan(x0, y0, dx=dx, dy=dy, rtol=rtol, executable=executable,
               outfile=outfile, evalsfile=evalsfile, pphw=pphw,
               xml=xml, template=template, interactive=interactive, 
               **waveguide_params)
    J.solve()


if __name__ == '__main__':
    find_EP.__doc__ = Jordan.__doc__
    argh.dispatch_command(find_EP)
