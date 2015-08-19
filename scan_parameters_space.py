#!/usr/bin/env python2.7

import os
import numpy as np
import shutil
import subprocess
import sys

import argh

import bloch
from ep.waveguide import Dirichlet
from helper_functions import replace_in_file


def run_code():
    if os.environ.get('TMPDIR') and os.environ.get('NSLOTS'):
        print "running code on cluster..."
        print "$TMPDIR", os.environ.get('TMPDIR')
        print "$NSLOTS", os.environ.get('NSLOTS')
        cmd = ("time mpirun -np {SLURM_NTASKS} "
               "solve_xml_mumps_dev").format(**os.environ)
    else:
        print "running code locally..."
        cmd = "solve_xml_mumps_dev"
    subprocess.call(cmd.split())


def print_diff_warning(array, name):
    """Print a warning if consecutive values in the input array differ by
    less than 1."""
    if np.any(np.diff(array) < 1):
        print """
            WARNING: not all {1} values can be resolved.

            diff(array) = {0}
        """.format(np.diff(array), name)


def archive(infile, outfile, delete=True):
    """Copy infile to outfile, compress the destination file and remove the
    source."""
    try:
        shutil.copy(infile, outfile)
        subprocess.call(['gzip', outfile])
        if delete:
            os.remove(infile)
    except:
        print "WARNING: could not archive " + infile


@argh.arg("--eps", type=float, nargs="+")
@argh.arg("--delta", type=float, nargs="+")
def raster_eps_delta(N=2.6, pphw=300, eta=0.1, W=1.0, xml="input.xml",
                     xml_template="input.xml_template", eps=[0.01, 0.1, 30],
                     delta=[0.3, 0.7, 50], dryrun=False):

    # k_x for modes 0 and 1
    k0, k1 = [np.sqrt(N**2 - n**2)*np.pi for n in (1, 2)]
    kr = k0 - k1

    # ranges
    eps_range = np.linspace(*eps)
    delta_range = np.linspace(*delta)
    print "eps_range", eps_range
    print "delta_range", delta_range

    # check if eps/delta values can be resolved
    r_ny_eps = (eps_range*(N*pphw+1)).astype(int)
    r_nx_L = (abs(2.*np.pi/(kr + delta_range))*(N*pphw+1)).astype(int)
    print "grid-points per full amplitude eps", r_ny_eps
    print "grid-points per length L", r_nx_L

    print_diff_warning(r_ny_eps, "epsilon")
    print_diff_warning(r_nx_L, "length")

    if dryrun:
        sys.exit()

    def update_boundary(eps, delta):
        L = abs(2*np.pi/(kr + delta))

        # choose discretization such that r_nx < len(x_range)
        # r_nx_L = (abs(2*np.pi/(kr + delta))*(N*pphw + 1)).astype(int)
        r_nx_L = (L(N*pphw + 1)).astype(int)
        x_range = np.linspace(0, L, r_nx_L)
        WG = Dirichlet(loop_type='Constant', N=N, L=L, W=W, eta=eta)

        xi_lower, xi_upper = WG.get_boundary(x=x_range, eps=eps, delta=delta)
        print "lower.boundary.shape", xi_lower.shape

        np.savetxt("lower.boundary", zip(x_range, xi_lower))
        np.savetxt("upper.boundary", zip(x_range, xi_upper))

        N_file_boundary = len(x_range)
        replacements = {'LENGTH': str(L),
                        'WIDTH': str(W),
                        'MODES': str(N),
                        'PPHW': str(pphw),
                        'GAMMA0': str(eta),
                        'NEUMANN': "0",
                        'N_FILE_BOUNDARY': str(N_file_boundary),
                        'BOUNDARY_UPPER': 'upper.boundary',
                        'BOUNDARY_LOWER': 'lower.boundary'}

        replace_in_file(xml_template, xml, **replacements)

    # parameters, eigenvalues and eigenvectors
    # eps, delta, ev0, ev1, overlap = [ [] for n in range(5) ]
    eps, delta, ev0, ev1, overlap = []*5
    tmp = "bloch.tmp"
    for e in eps_range:
        for d in delta_range:
            update_boundary(e, d)
            run_code()
            try:
                # bloch_evals = bloch.get_eigensystem()
                # TODO: why column 0 and not 1 to access right moving modes?
                # bloch_evals = np.array(bloch_evals)[0, :2]
                # if bloch.get_eigensystem is not called with modes, dx, etc.,
                # these values are read from the xml file
                bloch_evals, _, bloch_evecs, _ = bloch.get_eigensystem(return_eigenvectors=True)
                bloch_evals, bloch_evecs = [np.array(x)[:2] for x in (bloch_evals, bloch_evecs)]
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
                    evals_file = "evals_eps_{:.8f}_delta_{:.8f}.dat".format(e, d)
                    archive("Evals.complex_potential.dat", evals_file)
                    os.remove("Evecs.complex_potential.dat")
                    os.remove("Evecs.complex_potential.abs")
                    xml_file = "xml_eps_{:.8f}_delta_{:.8f}.dat".format(e, d)
                    archive("input.xml", xml_file, delete=False)
            except:
                print "Evals, evecs or xml file not found!"
            # tmp.out is not written if job is part of a job-array
            tmp_file = "tmp_eps_{:.8f}_delta_{:.8f}.out".format(e, d)
            archive("tmp.out", tmp_file)
            # be backwards compatible in case no jpg is written
            jpg_file = "jpg_eps_{:.8f}_delta_{:.8f}.jpg".format(e, d)
            archive("pic.geometry.complex_potential.1.jpg", jpg_file)

    eps, delta, ev0, ev1, overlap = [np.array(x) for x in (eps, delta, ev0,
                                                           ev1, overlap)]
    np.savetxt("bloch_modes.dat", zip(eps, delta,
                                      ev0.real, ev0.imag,
                                      ev1.real, ev1.imag, overlap))

if __name__ == '__main__':
    argh.dispatch_command(raster_eps_delta)
