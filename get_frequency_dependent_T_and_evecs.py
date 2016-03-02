#!/usr/bin/env python

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

import argh


def propagate_back(T, W=0.05, N=2.6, l=1.0, eta=0.0):
    k = N*np.pi/W
    k1, k2 = [np.sqrt(k**2 - (n*np.pi/W)**2) for n in (1, 2)]

    phase_k1 = (k1 + 1j*eta/2.*k/k1)*l
    phase_k2 = (k2 + 1j*eta/2.*k/k2)*l
    T12 = np.array([[np.exp(1j*phase_k1), 0],
                    [0, np.exp(1j*phase_k2)]])
    T21 = T12

    return scipy.linalg.inv(T12).dot(T).dot(scipy.linalg.inv(T21))


def get_eigenvalues(input_file=None, frequency_file=None, evecs_file=None, l=0.0, eta=0.0):
    """docstring for get_eigenvalues"""
    # config 0: l=
    # config 1: l=
    # config 2: l=0.395
    # config 3: l=0.47925
    # config 4: l=

    print "distance cavity - antenna:", l
    print "eta:", eta

    f = np.load(frequency_file)
    T = np.load(input_file)
    # print "f.shape", f.shape
    # print "T.shape", T.shape

    data = []
    matrix_data = []
    for n, Tn in enumerate(T):
        fn = f[n]
        Tn_flat = np.concatenate([[an.real, an.imag] for an in Tn.flatten()])
        matrix_data.append(np.concatenate([[fn], Tn_flat]))
        Tn = propagate_back(Tn, l=l, eta=eta)
        eigenvalues, eigenstates = scipy.linalg.eig(Tn.T)

        # sort eigenvalues and eigenstates
        sort_idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[..., sort_idx]
        eigenstates = eigenstates[..., sort_idx]
        modes = len(eigenvalues)
        v1, v2 = [eigenstates[m, 0] for m in (0, 1)]
        c1, c2 = [eigenstates[m, 1] for m in (0, 1)]

        data.append([v1, v2, c1, c2])

        if np.isclose(fn, 7.8):
            print "arctan(|v1/v2|)", np.arctan(np.abs(v1/v2))
            print "arctan(|c1/c2|)", np.arctan(np.abs(c1/c2))
            print "phase(v1/v2)", np.angle(v1/v2)
            print "phase(c1/c2)", np.angle(c1/c2)

            np.savetxt(input_file.replace(".npy", "_7.8GHz.dat"), Tn)
            if evecs_file:
                with open(evecs_file, "w") as file:
                    for _ in range(1):
                        for n in range(modes):
                            file.write('{} {} '.format(np.abs(eigenvalues[n]),
                                                    np.angle(eigenvalues[n])))
                            for m in range(modes):
                                v = eigenstates[m, n]
                                file.write('{v.real} {v.imag} '.format(v=v))
                        file.write('\n')

    v1, v2, c1, c2 = np.array(data).T

    # matrix_data = np.array(matrix_data)
    # np.savetxt(input_file.replace(".npy", "_matrix_data.dat"),
    #            matrix_data)
    # np.savetxt(input_file.replace(".npy", "_arctan_data.dat"),
    #            zip(np.arctan(abs(v1/v2)), np.arctan(abs(c1/c2))))

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(9, 8))

    plt.suptitle(r"Distance cavity - antenna: {}m".format(l))

    ax0.plot(f, np.abs(T[:,0,0])**2, "r-", label=r"$|t_{11}|^2$")
    ax0.plot(f, np.abs(T[:,1,1])**2, "b-", label=r"$|t_{22}|^2$")
    ax0.plot(f, np.abs(T[:,0,1])**2, "g-", label=r"$|t_{12}|^2$")
    ax0.plot(f, np.abs(T[:,1,0])**2, "-", color="orange", label=r"$|t_{21}|^2$")
    ax0.legend(loc="upper left")

    ax1.plot(f, np.arctan(abs(v1/v2)), "r-")
    ax1.plot(f, np.arctan(abs(c1/c2)), "b-")

    ax2.plot(f, np.angle(v1/v2), "r-")
    ax2.plot(f, np.angle(c1/c2), "b-")

    ax1.set_ylim(0, np.pi/2)

    ax0.set_ylabel(r"Transmission intensity")
    ax1.set_ylabel(r"$\arctan(|c_1/c_2|)$")
    ax2.set_xlabel(r"Frequency $\nu$ (GHz)")
    ax2.set_ylabel(r"$\operatorname{Arg}(c_1/c_2)$")

    for ax in fig.get_axes():
        ax.set_xlim(7, 8.8)
        ax.axvline(x=7.8, color="k")

    plt.show()


if __name__ == '__main__':
    argh.dispatch_command(get_eigenvalues)
