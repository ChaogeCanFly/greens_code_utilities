#!/usr/bin/env python2.7

import fileinput
import matplotlib.pyplot as plt
import numpy as np
import re
import subprocess

import os
import shutil
import sys

import argh

from ep.waveguide import Waveguide
from xmlparser import XML


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


def get_Bloch_eigenvalues(xml='input.xml', evalsfile='Evals.sine_boundary.dat',
                          modes=None, dx=None, r_nx=None, sort=True, 
                          fold_back=True, return_velocities=False):
    """Extract the eigenvalues beta and return the Bloch eigenvalues.

        Parameters:
        -----------
            xml: str
                Input xml file.
            evalsfile: str
                Eigenvalues input file.
            dx: float
                Grid spacing.
            r_nx: int
                Grid dimension in x-direction.
            fold_back: bool
                Whether to fold back the Bloch eigenvalues into the 1. BZ.
            return_velocities: bool
                Whether to return group velocities.

        Returns:
        --------
            k_left, k_right: ndarrays
                Bloch eigenvalues of left and right movers.
            v_left, v_right: ndarrays (optional)
                Velocities of left and right movers.
    """

    if modes is None or dx is None or r_nx is None:
        params = XML(xml).params
        modes = params.get("modes")
        dx = params.get("dx")
        r_nx = params.get("r_nx")

    k0, k1 = [ np.sqrt(modes**2 - n**2)*np.pi for n in (0,1) ]
    kr = k0 - k1

    print "modes", modes
    print "k0", k0
    print "k1", k1
    print "kr", kr
    print "dx:", dx
    print "r_nx:", r_nx
    print "dx*r_nx:", dx*r_nx

    beta, velocities = np.genfromtxt(evalsfile, unpack=True,
                                     usecols=(0, 1), dtype=complex,
                                     converters={0: convert_to_complex,
                                                 1: convert_to_complex})

    k = np.angle(beta) - 1j*np.log(np.abs(beta))
    k /= dx*r_nx
    k_left = k[:len(k)/2]
    k_right = k[len(k)/2:]

    v_left = velocities[:len(k)/2]
    v_right = velocities[len(k)/2:]

    if sort:
        sort_mask = np.argsort(abs(k_left.imag))
        k_left = k_left[sort_mask]
        v_left = v_left[sort_mask]
        sort_mask = np.argsort(abs(k_right.imag))
        k_right = k_right[sort_mask]
        v_right = v_right[sort_mask]

    if fold_back:
        k_left = np.mod(k_left.real, kr) + 1j*k_left.imag
        k_right = np.mod(k_right.real, kr) + 1j*k_right.imag

    if return_velocities:
        return k_left, k_right, v_left, v_right
    else:
        return k_left, k_right


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
            template: str
                Input .xml template name.
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
                 template='input.xml_template',
                 xml='input.xml', **waveguide_params):

        self.values = [[x0, y0], [x0+dx, y0+dx]]
        self.evals = []
        self.rtol = rtol
        self.residual = 1.

        self.waveguide_params = waveguide_params

        self.executable = executable
        self.datafile = datafile
        self.xml = xml
        self.template = template
        self._update_boundary(x0, y0)
        print XML(xml).params
        self.__dict__.update(XML(xml).params)
        self.evalsfile = evalsfile


    def _run_code(self):
        params = {'E': self.executable,
                  'I': self.xml}
        cmd = "subSGE.py -l -e {E} -i {I}".format(**params)
        subprocess.check_call(cmd, shell=True)

    def _iterate(self):
        (x0, y0), (x1, y1) = self.values[-2:]
        dx, dy = x1-x0, y1-y0

        print "x0, x1", x0, x1
        print "y0, y1", y0, y1
        print "dx, dy", dx, dy

        eigenvalues = []
        parameters = [ (n,m) for n in x0, x1 for m in y0, y1 ]
        for x, y in parameters:
            print "n,m", x, y
            self._update_boundary(x, y)
            self._run_code()
            bloch_modes = get_Bloch_eigenvalues(evalsfile=self.evalsfile,
                                                dx=self.dx, r_nx=self.r_nx)
            # take the first two right-movers
            # k1 -> k1 mod kr
            bloch_modes = np.array(bloch_modes)[1,:2]
            print "bloch_modes", bloch_modes
            eigenvalues.append(bloch_modes)
            print "eigenvalues[-1]", eigenvalues[-1]

        #print eigenvalues
        e1, e2 = eigenvalues[-1]
        evals = np.asarray(eigenvalues).T.flatten()
        self.evals.append(eigenvalues)
        print "evals", evals


        gradient = np.array([[0, 1, 0, -1, 0, -1, 0, 1],
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

        N = self.waveguide_params.get("N")
        eta = self.waveguide_params.get("eta")

        k0, k1 = [ np.sqrt(N**2 - n**2)*np.pi for n in 0, 1 ]
        L = abs(2*np.pi/(k0 - k1 + y))

        WG = Waveguide(N=N, L=L, loop_type='Constant')
        self.WG = WG

        WG.x_EP = x
        WG.y_EP = y

        xi_lower, xi_upper = WG.get_boundary(eps=x, delta=y)

        np.savetxt("lower.profile", zip(WG.t, xi_lower))
        np.savetxt("upper.profile", zip(WG.t, xi_upper))
        # np.savetxt("lower_x_{}_y_{}.profile".format(x,y), zip(self.WG.t, xi_lower))
        # np.savetxt("upper_x_{}_y_{}.profile".format(x,y), zip(self.WG.t, xi_upper))

        # update .xml

        N_file = len(WG.t)
        replacements = {
                'L"> L':             'L"> {}'.format(L),
                'N_file"> N_file':   'N_file"> {}'.format(N_file),
                'Gamma0"> Gamma0':   'Gamma0"> {}'.format(eta)
                }

        with open(self.template) as src_xml:
            src_xml = src_xml.read()

        for src, target in replacements.iteritems():
            src_xml = src_xml.replace(src, target)

        out_xml = os.path.abspath(self.xml)
        with open(out_xml, "w") as out_xml:
            out_xml.write(src_xml)


    def solve(self):
        #while self.residual > self.rtol:
        for n in range(5):
            xi, yi = self._iterate()
            self.values.append([xi, yi])
            #self._update_boundary(xi, yi)
            print "residual", self.residual
            print "rtol", self.rtol
            print "x, y:", self.values[-1]
            print "x_EP, y_EP = ", self.WG.x_EP, self.WG.y_EP
            prompt = raw_input("Continue? (y)es/(n)o/(p)lot ")

            if not prompt:
                pass
            elif prompt[0] == 'n':
                print self.values
                np.savetxt(self.datafile, self.values)
                sys.exit()
            elif prompt[0] == 'p':
                f = plt.figure(0)
                for n, (x, y) in enumerate(self.values):
                    plt.plot(x, y, "ro")
                    plt.text(x, y, str(n), fontsize=12)
                plt.show()
            else:
                pass

        print self.values
        np.savetxt(self.datafile, self.values)


@argh.arg('x0', type=float)
@argh.arg('y0', type=float)
@argh.arg('-N', '--N', type=float)
@argh.arg('--eta', type=float)
def find_EP(x0, y0, dx=1e-2, dy=1e-2, rtol=1e-6, executable='solve_xml_mumps', 
            datafile='jordan.out', evalsfile='Evals.sine_boundary.dat',
            xml='input.xml', template='input.xml_template', **waveguide_params):
    J = Jordan(x0, y0, dx=dx, dy=dy, rtol=rtol, executable=executable,
               datafile=datafile, evalsfile=evalsfile, xml=xml,
               template=template, **waveguide_params)
    J.solve()


if __name__ == '__main__':
    find_EP.__doc__ = Jordan.__doc__
    argh.dispatch_command(find_EP)
