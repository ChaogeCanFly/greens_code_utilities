#!/usr/bin/env python2.7

import argh
import subprocess
from ep.waveguide import Waveguide
import numpy as np
import re


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
            datafile: str
                Parameter output file.
            evalsfile: str
                Eigenvalue file.
            xml: str
               Input .xml file name.
            kwargs:
                Parameters of the ep.waveguide.Waveguide class.

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
                 executable='solve_xml_mumps', datafile='jordan.out',
                 evalsfile='Evals.sine_boundary.dat',
                 xml='input.xml', **kwargs):

        self.values = [[x0, y0], [x0+dx, y0+dx]]
        self.evals = []
        self.rtol = rtol
        self.residual = None
        self.dx = None

        self.executable = executable
        self.datafile = datafile
        self.xml = xml
        self.evalsfile = evalsfile

        self._get_dx()
        self.WG = Waveguide(**kwargs)

    def _run_code(self):
        params = {'E': self.executable,
                  'I': self.xml}
        cmd = "subSGE.py -l -e {E} -i {I}".format(**params)
        subprocess.check_call(cmd, shell=True)

    def _get_dx(self):
        with open(self.xml) as f:
            for line in f.readlines():
                if "points_per_halfwave" in line:
                    pph = re.split("[><]", line)[-3]
                    dx = 1./(float(pph) + 1)
        self.dx = dx

    @classmethod
    def _get_eigenvalues(self, evalsfile=None, dx=None):
        if not evalsfile:
            evalsfile = self.evalsfile
        if not dx:
            dx = self.dx
        beta = np.genfromtxt(evalsfile, unpack=True, dtype=complex, usecols=0,
                             converters={0: lambda s: convert_to_complex(s)})
        k = np.angle(beta)/dx
        k_l = k[:len(k)/2]
        k_r = k[len(k)/2:]

        return k_l[0], k_r[0]

    def _iterate(self):
        (x0, y0), (x1, y1) = self.values[-2:]
        dx, dy = x1-x0, y1-y0

        parameters = [ (n,m) for n in x0, x1 for m in y0,y1 ]
        for x, y in parameters:
            self._run_code()
            e00, e01, e10, e11 = self._get_eigenvalues(x,y)
            self.evals.append([e00, e01, e10, e11])

        vx = (e10[0]-e10[1]) - (e00[0]-e00[1])
        vx /= dx
        vy = (e01[0]-e01[1]) - (e00[0]-e00[1])
        vy /= dy

        norm = np.sqrt(vx**2 + vy**2)
        res_x = vx/norm * (e00[0] - e00[1])
        res_y = vy/norm * (e00[0] - e00[1])

        self.residual = np.sqrt(res_x**2 + res_y**2)

        x2 = x1 - res_x
        y2 = y1 - res_y

        return x2, y2

    def _update_boundary(self, x, y):
        xi_lower, xi_upper = self.WG.get_boundary(eps=x, delta=y)
        np.savetxt("lower.profile", zip(WG.t, xi_lower))
        np.savetxt("upper.profile", zip(WG.t, xi_upper))

    def solve(self):
        while self.residual < self.rtol:
            print "x, y:", self.values[-1]
            xi, yi = self._iterate()
            self.values.append([xi,yi])
            self._update_boundary(xi, yi)

        print self.values
        np.savetxt(self.datafile, self.values)


def convert_to_complex(s):
    """Convert a string of the form (x,y) to a complex number z = x+1j*y.

        Parameters:
        -----------
            s: str

        Returns:
        -------
            z: complex
    """

    regex = re.compile(r'\(([^,\)]+),([^,\)]+)\)')
    x, y = map(float, regex.match(s).groups())
    return x + 1j*y


def find_EP(*args, **kwargs):
    J = Jordan(*args, **kwargs)
    J.solve()


if __name__ == '__main__':
    argh.dispatch_command(find_EP)
