#!/usr/bin/env python2.7

import numpy as np
import re
import subprocess

import argh
import xml.etree.ElementTree as ET

from ep.waveguide import Waveguide


class XML(object):
    """Simple wrapper class for xml.etree.ElementTree.
    
        Parameters:
        -----------
            xml: str
                Input xml file.
                
        Attributes:
        -----------
            root: Element object
            params: dict
                Dictionary of the parameters parsed from the input xml.
    """

    def __init__(self, xml):
        self.xml = xml
        self.root = ET.parse(xml).getroot()
        self.params = self._read_xml()

    def _read_xml(self):
        """Read all variables from the input.xml file."""

        params = {}
        for elem in self.root.iter(tag='param'):
            name = elem.attrib.get('name')
            value = elem.text
            try:
                params[name] = float(value)
            except ValueError:
                pass
                # params[name] = value

        self.__dict__.update(params)

        self.nyout = self.modes*self.points_per_halfwave
        self.dx = self.W/(self.nyout + 1.)
        self.dy = self.dx
        self.r_nx = int(self.W/self.dx)
        self.r_ny = int(self.L/self.dy)

        params.update(self.__dict__)

        return params


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
            waveguide_params:
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
                 xml='input.xml', **waveguide_params):

        self.values = [[x0, y0], [x0+dx, y0+dx]]
        self.evals = []
        self.rtol = rtol
        self.residual = 1.

        self.executable = executable
        self.datafile = datafile
        self.xml = xml
        self.xml_params = XML(xml).params
        self.evalsfile = evalsfile

        self.dx = self.xml_params.get("dx")
        print "self.dx", self.dx
        self.r_nx = self.xml_params.get("r_nx")
        self.modes = self.xml_params.get("modes")
        self.WG = Waveguide(**waveguide_params)

    def _run_code(self):
        params = {'E': self.executable,
                  'I': self.xml}
        cmd = "subSGE.py -l -e {E} -i {I}".format(**params)
        subprocess.check_call(cmd.split())

    @classmethod
    def get_eigenvalues(self, evalsfile=None, dx=None, r_nx=None, sort=True):
        if evalsfile is None:
            evalsfile = self.evalsfile
        if dx is None:
            dx = self.dx
        if r_nx is None:
            r_nx = self.r_nx
            r_nx = 1.

        beta, velocities = np.genfromtxt(evalsfile, unpack=True,
                                         usecols=(0, 1), dtype=complex,
                                         converters={0: convert_to_complex})
        k = np.angle(beta) - 1j*np.log(np.abs(beta))
        k /= r_nx*dx
        k_left = k[:len(k)/2]
        k_right = k[len(k)/2:]

        if sort:
            sort_mask = np.argsort(abs(k_left.imag))
            k_left = k_left[sort_mask]
            sort_mask = np.argsort(abs(k_right.imag))
            k_right = k_right[sort_mask]

        return k_left, k_right

    def _iterate(self):
        (x0, y0), (x1, y1) = self.values[-2:]
        dx, dy = x1-x0, y1-y0

        print "x0, x1", x0, x1
        print "y0, y1", y0, y1
        print "dx, dy", dx, dy

        eigenvalues = []
        parameters = [ (n,m) for n in x0, x1 for m in y0, y1 ]
        for x, y in parameters:
            # print "n,m", x, y
            self._update_boundary(x, y)
            self._run_code()
            eigenvalues.append(self.get_eigenvalues(x, y))

        e1, e2 = eigenvalues[-1]
        evals = np.asarray(eigenvalues).T.flatten()
        self.evals.append(eigenvalues)

        gradient =  np.array([[0, 1, 0, -1, 0, -1, 0, 1],
                              [0, 0, 1, -1, 0, 0, -1, 1]])

        gradient_x, gradient_y = gradient.dot(evals)
        gradient_x /= dx
        gradient_y /= dy

        delta = e2-e1

        # norm = (gradient_x**2 + gradient_y**2)
        norm = (np.abs(gradient_x)**2 + 
                np.abs(gradient_y)**2)

        res_x = gradient_x/norm * delta
        res_y = gradient_y/norm * delta

        self.residual = np.sqrt(np.abs(res_x)**2 + np.abs(res_y)**2)

        print "res_x", res_x
        print "res_y", res_y

        x2 = x1 - res_x
        y2 = y1 - res_y

        return x2, y2

    def _update_boundary(self, x, y):
        x, y = [ np.real(n) for n in x, y ]
        xi_lower, xi_upper = self.WG.get_boundary(eps=x, delta=y)
        np.savetxt("lower.profile", zip(self.WG.t, xi_lower))
        np.savetxt("upper.profile", zip(self.WG.t, xi_upper))

    def solve(self):
        #while self.residual > self.rtol:
        for n in range(5):
            print "residual", self.residual
            print "rtol", self.rtol
            print "x, y:", self.values[-1]
            xi, yi = self._iterate()
            self.values.append([xi, yi])
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
            z: complex float
    """

    regex = re.compile(r'\(([^,\)]+),([^,\)]+)\)')
    x, y = map(float, regex.match(s).groups())
    return x + 1j*y


def find_EP(*args, **kwargs):
    J = Jordan(*args, **kwargs)
    J.solve()


if __name__ == '__main__':
    argh.dispatch_command(find_EP)
