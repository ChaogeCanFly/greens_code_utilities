#!/usr/bin/env python2.7

import os
import numpy as np
import shutil
import subprocess
import sys

import argh

import bloch
from ep.waveguide import Waveguide
from helpers import replace_in_file


def run_code(local=False):
    print "running code..."
    if local:
        cmd = "solve_xml_mumps"
    else:
        print "$TMPDIR", os.environ.get('TMPDIR')
        print "$NSLOTS", os.environ.get('NSLOTS')
        cmd = "mpirun -machinefile {TMPDIR}/machines -np {NSLOTS} solve_xml_mumps".format(**os.environ)
    subprocess.call(cmd.split())


@argh.arg("--eps", type=float, nargs="+")
@argh.arg("--delta", type=float, nargs="+")
def raster_eps_delta(N=1.05, pphw=300, eta=0.1, xml="input.xml", local=False,
                     xml_template="input.xml_template", eps=[0.01, 0.1, 30],
                     delta=[0.3, 0.7, 50], dryrun=False):

    # k_x for modes 0 and 1
    k0, k1 = [ np.sqrt(N**2 - n**2)*np.pi for n in 0, 1 ]
    kr = k0 - k1

    # ranges
    eps_range = np.linspace(*eps)
    delta_range = np.linspace(*delta)
    print "eps_range", eps_range
    print "delta_range", delta_range
    print "len(eps)*len(delta)", len(eps_range)*len(delta_range)

    # check if eps/delta values can be resolved
    r_ny_eps = (eps_range*(N*pphw+1)).astype(int)
    r_nx_L = (abs(2*np.pi/(kr + delta_range))*(N*pphw+1)).astype(int)

    print "grid-points per full amplitude eps", r_ny_eps
    print "grid-points per length L", r_nx_L

    if np.any(np.diff(r_ny_eps) < 1):
        print """
            WARNING: not all epsilon values can be resolved.

            diff(r_ny_eps) = {0}
        """.format(np.diff(r_ny_eps))

    if np.any(abs(np.diff(r_nx_L)) < 1):
        print """
            WARNING: not all L values can be resolved.

            diff(r_nx_L) = {0}
        """.format(abs(np.diff(r_nx_L)))

    if dryrun:
        sys.exit()

    def update_boundary(x, y):
        L = abs(2*np.pi/(k0 - k1 + y))

        WG = Waveguide(L=L, loop_type='Constant', N=N, eta=eta)
        WG.x_EP = x
        WG.y_EP = y

        xi_lower, xi_upper = WG.get_boundary(eps=x, delta=y)

        np.savetxt("lower.profile", zip(WG.t, xi_lower))
        np.savetxt("upper.profile", zip(WG.t, xi_upper))

        N_file = len(WG.t)
        replacements = {'L"> L':             'L"> {}'.format(L),
                        'wave"> pphw':       'wave"> {}'.format(pphw),
                        'N_file"> N_file':   'N_file"> {}'.format(N_file),
                        'Gamma0"> Gamma0':   'Gamma0"> {}'.format(eta)}

        replace_in_file(xml_template, xml, **replacements)

    eps, delta, ev0, ev1 = [ [] for n in range(4) ]
    tmp = "bloch.tmp"
    for e in eps_range:
        for d in delta_range:
            update_boundary(e, d)
            run_code(local=local)
            bloch_modes = bloch.get_eigensystem()
            # TODO: why column 0 and not 1 to access the right moving modes?
            bloch_modes = np.array(bloch_modes)[0, :2]
            ev0.append(bloch_modes[0])
            ev1.append(bloch_modes[1])
            eps.append(e)
            delta.append(d)
            with open(tmp, "a") as f:
                f.write("{} {} {} {} {} {}\n".format(e, d, bloch_modes[0].real,
                                                     bloch_modes[0].imag,
                                                     bloch_modes[1].real,
                                                     bloch_modes[1].imag))
            # backup output files
            try:
                evals_file = "evals_eps_{:.8f}_delta_{:.8f}.dat".format(e, d)
                shutil.copy("Evals.sine_boundary.dat", evals_file)
                subprocess.call(['gzip', evals_file])
                evecs_file = "evecs_eps_{:.8f}_delta_{:.8f}.dat".format(e, d)
                shutil.copy("Evecs.sine_boundary.dat", evecs_file)
                subprocess.call(['gzip', evecs_file])
                xml_file = "xml_eps_{:.8f}_delta_{:.8f}.dat".format(e, d)
                shutil.copy("input.xml", xml_file)
                subprocess.call(['gzip', xml_file])
            except:
                pass
            if not local:
                # tmp.out is not written if job is part of a job-array
                try:
                    tmp_file = "tmp_eps_{:.8f}_delta_{:.8f}.out".format(e, d)
                    shutil.copy("tmp.out", tmp_file)
                    subprocess.call(['gzip', tmp_file])
                except:
                    pass
            try:
                jpg_file = "jpg_eps_{:.8f}_delta_{:.8f}.jpg".format(e, d)
                shutil.copy("pic.geometry.sine_boundary.1.jpg", jpg_file)
                subprocess.call(['gzip', jpg_file])
            except:
                pass

    eps, delta, ev0, ev1 = [ np.array(x) for x in eps, delta, ev0, ev1 ]

    np.savetxt("bloch_modes.dat", zip(eps, delta,
                                      ev0.real, ev0.imag,
                                      ev1.real, ev1.imag))

if __name__ == '__main__':
    argh.dispatch_command(raster_eps_delta)
