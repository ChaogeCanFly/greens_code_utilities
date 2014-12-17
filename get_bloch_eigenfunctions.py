#!/usr/bin/env python2.7

import glob
import numpy as np
import os

import argh

import bloch
from helper_functions import natural_sorting
from xmlparser import XML


def get_eigenvectors(folder='folder'):
    os.chdir(folder)
    with open("bloch_all_vectors.tmp", "w") as f:
        for evecs in sorted(glob.glob("_eps*/evecs*")):
            print evecs
            eps, delta = [ os.path.basename(evecs).split("_")[n] for n in 2, 4 ]
            eps, delta = [ float(x) for x in eps, delta.replace(".dat.gz", "") ]

            xml = evecs.replace("evecs", "xml")
            # xml_params = XML(xml).params

            evals, _, evecs, _ = bloch.get_eigensystem(xml=xml,
                                                       evalsfile=evecs.replace("evecs", "evals"),
                                                       evecsfile=evecs,
                                                       return_eigenvectors=True,
                                                       modes=1.05)
            evals, evecs = [ np.array(x)[:2] for x in evals, evecs ]
            ev0, ev1 = evals
            v0, v1 = evecs
            overlap = (np.abs(v0-v1)**2).sum()
            # f.write("{:>18} {:>18} {:>18} {:>18} {:>18} {:>18}\n".format(eps, delta, ev0.real, ev0.imag, ev1.real, ev1.imag))
            f.write("{:>18} {:>18} {:>18} {:>18} {:>18} {:>18}\n".format(eps, delta, overlap, overlap, overlap, overlap))


if __name__ == '__main__':
    argh.dispatch_command(get_eigenvectors)
