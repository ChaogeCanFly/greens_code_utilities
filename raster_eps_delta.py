#!/usr/bin/env python2.7

import os
import numpy as np
import shutil
import subprocess
import sys

import argh

import bloch
from ep.waveguide import Waveguide
from helper_functions import replace_in_file


def run_code():
    if os.environ.get('TMPDIR') and os.environ.get('NSLOTS'):
        print "running code on cluster..."
        print "$TMPDIR", os.environ.get('TMPDIR')
        print "$NSLOTS", os.environ.get('NSLOTS')
        cmd = ("mpirun -machinefile {TMPDIR}/machines -np {NSLOTS} "
               "solve_xml_mumps_dev").format(**os.environ)
    else:
        print "running code locally..."
        cmd = "solve_xml_mumps"

    subprocess.call(cmd.split())


@argh.arg("--eps", type=float, nargs="+")
@argh.arg("--delta", type=float, nargs="+")
def raster_eps_delta(N=1.05, pphw=300, eta=0.1, xml="input.xml",
                     xml_template="input.xml_template", eps=[0.01, 0.1, 30],
                     delta=[0.3, 0.7, 50], dryrun=False, neumann=1):

    # k_x for modes 0 and 1
    if neumann:
        k0, k1 = [ np.sqrt(N**2 - n**2)*np.pi for n in 0, 1 ]
    else:
        k0, k1 = [ np.sqrt(N**2 - n**2)*np.pi for n in 1, 2 ]

    kr = k0 - k1

    # ranges
    eps_range = np.linspace(*eps)
    delta_range = np.linspace(*delta)
    print "eps_range", eps_range
    print "delta_range", delta_range

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

    def update_boundary(eps, delta):
        L = abs(2*np.pi/(k0 - k1 + delta))

        # choose discretization such that r_nx < len(x_range)
        r_nx_L = (abs(2*np.pi/(kr + delta))*(N*pphw + 1)).astype(int)
        x_range = np.linspace(0, L, r_nx_L)
        WG = Waveguide(L=L, loop_type='Constant', N=N, eta=eta, neumann=neumann)

        xi_lower, xi_upper = WG.get_boundary(x=x_range, eps=eps, delta=delta)
        print "xi_lower.shape", xi_lower.shape

        np.savetxt("lower.profile", zip(x_range, xi_lower))
        np.savetxt("upper.profile", zip(x_range, xi_upper))

        # N_file = len(WG.t)
        N_file = len(x_range)
        replacements = {'L"> L':             'L"> {}'.format(L),
                        'modes"> modes':     'modes"> {}'.format(N),
                        'wave"> pphw':       'wave"> {}'.format(pphw),
                        'N_file"> N_file':   'N_file"> {}'.format(N_file),
                        'neumann"> neumann': 'neumann"> {}'.format(neumann),
                        'upper_file"> file_upper': 'upper_file"> upper.profile',
                        'lower_file"> file_lower': 'lower_file"> lower.profile',
                        'Gamma0"> Gamma0':   'Gamma0"> {}'.format(eta)}

        replace_in_file(xml_template, xml, **replacements)

    # parameters, eigenvalues and eigenvectors
    eps, delta, ev0, ev1, overlap = [ [] for n in range(5) ]
    tmp = "bloch.tmp"
    for e in eps_range:
        for d in delta_range:
            update_boundary(e, d)
            run_code()
            try:
                ##bloch_evals = bloch.get_eigensystem()
                # TODO: why column 0 and not 1 to access the right moving modes?
                ##bloch_evals = np.array(bloch_evals)[0, :2]
                # if bloch.get_eigensystem is not called with modes, dx, etc.,
                # these values are read from the xml file
                bloch_evals, _, bloch_evecs, _ = bloch.get_eigensystem(return_eigenvectors=True, neumann=neumann)
                bloch_evals, bloch_evecs = [ np.array(x)[:2] for x in bloch_evals, bloch_evecs ]
                bloch_evecs_overlap = (np.abs(bloch_evecs[0]-bloch_evecs[1])**2).sum()
                print "overlap", bloch_evecs_overlap

                ev0.append(bloch_evals[0])
                ev1.append(bloch_evals[1])
                overlap.append(bloch_evecs_overlap)
                eps.append(e)
                delta.append(d)
                with open(tmp, "a") as f:
                    f.write("{} {} {} {} {} {}\n".format(e, d,
                                                            bloch_evals[0].real,
                                                            bloch_evals[0].imag,
                                                            bloch_evals[1].real,
                                                            bloch_evals[1].imag))
                    # backup output files
                    evals_file = "evals_eps_{:.8f}_delta_{:.8f}.dat".format(e,d)
                    shutil.copy("Evals.sine_boundary.dat", evals_file)
                    subprocess.call(['gzip', evals_file])
                    os.remove("Evecs.sine_boundary.dat")
                    os.remove("Evecs.sine_boundary.abs")
                    # evecs_file = "evecs_eps_{:.8f}_delta_{:.8f}.dat".format(e,d)
                    # shutil.copy("Evecs.sine_boundary.dat", evecs_file)
                    # subprocess.call(['gzip', evecs_file])
                    xml_file = "xml_eps_{:.8f}_delta_{:.8f}.dat".format(e,d)
                    shutil.copy("input.xml", xml_file)
                    subprocess.call(['gzip', xml_file])
            except:
                print "Evals, evecs or xml file not found!"
            # tmp.out is not written if job is part of a job-array
            try:
                tmp_file = "tmp_eps_{:.8f}_delta_{:.8f}.out".format(e, d)
                shutil.copy("tmp.out", tmp_file)
                subprocess.call(['gzip', tmp_file])
            except:
                print "Temp file not found!".format(tmp_file)
            # be backwards compatible in case no jpg is written
            try:
                jpg_file = "jpg_eps_{:.8f}_delta_{:.8f}.jpg".format(e, d)
                shutil.copy("pic.geometry.sine_boundary.1.jpg", jpg_file)
                subprocess.call(['gzip', jpg_file])
            except:
                print "JPG file not found!".format(jpg_file)

    eps, delta, ev0, ev1, overlap = [ np.array(x) for x in
                                       eps, delta, ev0, ev1, overlap ]
    np.savetxt("bloch_modes.dat", zip(eps, delta,
                                      ev0.real, ev0.imag,
                                      ev1.real, ev1.imag, overlap))

if __name__ == '__main__':
    argh.dispatch_command(raster_eps_delta)
