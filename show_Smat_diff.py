#!/usr/bin/env python2.7

import argh

@argh_hack()
def show_Smat_diff(folder_1=".", folder_2=".", datafile="Smat.sine_boundary.dat"):
    multi = np.loadtxt("{}/{}".format(folder_1, datafile), unpack=True, skiprows=4)
    single = np.loadtxt("{}/{}".format(folder_2, datafile), unpack=True, skiprows=4)

if __name__ == '__main__':
    argh.dispatch_command(show_Smat_diff)
