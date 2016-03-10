#!/usr/bin/env python

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

import argh

from T_Matrix import T_Matrix


def propagate_back(T, W=0.05, N=2.6, l=1.0, eta=0.0, f=None):
    if f:
        f *= 1e9  # frequency in GHz
        c = 299792458  # in m/s
        k = 2*np.pi*f/c
    else:
        k = N*np.pi/W

    k1, k2 = [np.sqrt(k**2 - (n*np.pi/W)**2) for n in (1, 2)]

    # if second mode not yet open
    if k < 2.*np.pi/W:
        k2 = 1j*np.sqrt((2.*np.pi/W)**2 - k**2)

    eta /= W

    phase_k1 = (k1 + 1j*eta/2.*k/k1)*l
    phase_k2 = (k2 + 1j*eta/2.*k/k2)*l
    T12 = np.array([[np.exp(1j*phase_k1), 0],
                    [0, np.exp(1j*phase_k2)]])
    T21 = T12

    return scipy.linalg.inv(T12).dot(T).dot(scipy.linalg.inv(T21))



def get_eigenstate_components(Tn):
    eigenvalues, eigenstates = scipy.linalg.eig(Tn)

    # sort eigenvalues and eigenstates
    sort_idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[..., sort_idx]
    eigenstates = eigenstates[..., sort_idx]
    modes = len(eigenvalues)
    v1, v2 = [eigenstates[m, 0] for m in (0, 1)]
    c1, c2 = [eigenstates[m, 1] for m in (0, 1)]

    return v1, v2, c1, c2, eigenvalues, eigenstates



@argh.arg("-l", type=float)
@argh.arg("-c", "--config", type=int)
@argh.arg("-m", "--matrix-output", type=str)
@argh.arg("-s", "--single-frequency", type=float)
def get_eigenvalues(input_file=None, frequency_file=None, evecs_file=None,
                    l=None, eta=0.0, config=None, exp=False, matrix_output=None,
                    plot=False, single_frequency=None):
    """docstring for get_eigenvalues"""

    # config 0: l=0.51576837377840601 (1 wavelength)
    # config 0: l=0.5157  (exp)       (1 wavelength)
    # config 1: l=0.49262250087499387 (2 wavelengths)
    # config 1: l=0.4926  (exp)       (2 wavelengths)
    # config 2: l=0.39514141195540475 (4 wavelengths)
    # config 2: l=0.3951  (exp)       (4 wavelengths)
    # config 3: l=0.4792450151610923  (4 wavelengths)
    # config 3: l=0.4793  (exp)       (4 wavelengths)
    # config 4: l=0.53112071257820737 (4 wavelengths)
    # config 4: l=0.5311   (exp)      (4 wavelengths)

    if single_frequency:
        input_file = "single_frequency_{}".format(single_frequency)

    if not single_frequency and not frequency_file:
        frequency_file = "frequency_" + input_file

    if not l:
        if config == 0:
            l = 0.51576837377840601
        elif config == 1:
            l = 0.49262250087499387
        elif config == 2:
            l = 0.39514141195540475
        elif config == 3:
            l = 0.4792450151610923
        elif config == 4:
            l = 0.53112071257820737
        else:
            l = 0.0

        if exp:
            if config == 0:
                l = 0.5157
            elif config == 1:
                l = 0.4926
            elif config == 2:
                l = 0.3951
            elif config == 3:
                l = 0.4793
            elif config == 4:
                l = 0.5311
            else:
                l = 0.0

    print
    print "distance cavity - antenna:", l
    print "eta:", eta
    print "input_file", input_file

    if not single_frequency:
        f = np.load(frequency_file)
        T = np.load(input_file)
    else:
        f = [single_frequency]
        T = [T_Matrix(transmission_matrix=True).t]

    T_original = T

    # convention here is like in experiment: t_nm is transmission amplitude
    # from mode n to mode m
    if not exp:
        T = np.asarray([Tn.T for Tn in T])
        print "WARNING: T-matrix is now transposed!"

    T_back_propagated = np.asarray([propagate_back(Tn, l=l, eta=eta, f=fn) for (Tn, fn) in zip(T, f)])

    # find index closest to target frequency
    if not single_frequency:
        n_target = (np.abs(f - 7.8)).argmin()
    else:
        n_target = 0

    data = []
    matrix_data_original = []
    matrix_data_backpropagated = []
    for n, Tn in enumerate(T_back_propagated):
        v1, v2, c1, c2, eigenvalues, eigenstates = get_eigenstate_components(Tn)
        data.append([v1, v2, c1, c2])

        # save output files
        Tn_flat_original = np.concatenate([[an.real, an.imag] for an in T[n, ...].flatten()])
        matrix_data_original.append(np.concatenate([[f[n]], Tn_flat_original]))
        Tn_flat_backpropagated = np.concatenate([[an.real, an.imag] for an in Tn.flatten()])
        matrix_data_backpropagated.append(np.concatenate([[f[n]], Tn_flat_backpropagated]))

        fn = f[n]
        if np.isclose(fn, f[n_target]):
            print
            print "@7.8GHz:", fn
            print "array_index", n
            print "arctan(|v1/v2|)", np.arctan(np.abs(v1/v2))
            print "arctan(|c1/c2|)", np.arctan(np.abs(c1/c2))
            print "phase(v1/v2)", np.angle(v1/v2)
            print "phase(c1/c2)", np.angle(c1/c2)
            print
            print "T-matrix (back propagated):"
            print Tn
            print np.abs(Tn)
            print "T-matrix (original):"
            print T_original[n_target]
            print np.abs(T_original[n_target])
            print
            print "evals"
            print eigenvalues[0]
            print np.abs(eigenvalues[0])
            print eigenvalues[1]
            print np.abs(eigenvalues[1])
            print "evecs"
            print eigenstates[:, 0]
            print eigenstates[:, 1]
            print

            modes = len(eigenvalues)
            # np.savetxt(input_file.replace(".npy", "_7.8GHz.dat"), Tn)
            if evecs_file:
                with open(evecs_file, "w") as file:
                    for n in range(modes):
                        file.write('{} {} '.format(np.abs(eigenvalues[n]),
                                                np.angle(eigenvalues[n])))
                        for m in range(modes):
                            v = eigenstates[m, n]
                            file.write('{v.real} {v.imag} '.format(v=v))
                    file.write('\n')

    v1, v2, c1, c2 = np.array(data).T

    if matrix_output:
        matrix_data_original = np.asarray(matrix_data_original)
        matrix_data_backpropagated = np.asarray(matrix_data_backpropagated)
        header = "frequency (in GHz), Re(t11), Im(t11), Re(t12), Im(t12), Re(t21), Im(t21), Re(t22), Im(t22)"
        np.savetxt(matrix_output.replace(".dat", "_original.dat"),
                   matrix_data_original, header=header)
        np.savetxt(matrix_output.replace(".dat", "_backpropagated.dat"),
                   matrix_data_backpropagated, header=header)

    # plot results
    fig, (ax0, ax00, ax1, ax2) = plt.subplots(nrows=4, figsize=(9, 10))

    plt.suptitle(r"Distance cavity - antenna: {}m ".format(l) + "\n"
                 + r"eta: {}".format(eta) + "\n"
                 + input_file)

    ax0.plot(f, np.abs(T_back_propagated[:,0,0])**2, "r-", label=r"$|t_{11}|^2$")
    ax0.plot(f, np.abs(T_back_propagated[:,1,1])**2, "b-", label=r"$|t_{22}|^2$")
    ax0.plot(f, np.abs(T_back_propagated[:,0,1])**2, "g-", label=r"$|t_{12}|^2$")
    ax0.plot(f, np.abs(T_back_propagated[:,1,0])**2, "-", color="orange", label=r"$|t_{21}|^2$")
    ax0.legend(loc="upper left")

    ax00.plot(f, np.angle(T_back_propagated[:,0,0]), "r-", label=r"$\operatorname{Arg}(t_{11})$")
    ax00.plot(f, np.angle(T_back_propagated[:,1,1]), "b-", label=r"$\operatorname{Arg}(t_{22})$")
    ax00.plot(f, np.angle(T_back_propagated[:,0,1]), "g-", label=r"$\operatorname{Arg}(t_{12})$")
    ax00.plot(f, np.angle(T_back_propagated[:,1,0]), "-", color="orange", label=r"$\operatorname{Arg}(t_{21})$")
    ax00.legend(loc="upper left")

    ax1.plot(f, np.arctan(abs(v1/v2)), "r-")
    ax1.plot(f, np.arctan(abs(c1/c2)), "b-")

    ax2.plot(f, np.angle(v1/v2), "r-")
    ax2.plot(f, np.angle(c1/c2), "b-")

    # for ax in (ax1, ):
    #     ax_yticks = np.arange(0, .75, 0.25)
    #     ax_yticklabels = [r"$0$", r"$+\frac{\pi}{4}$", r"$\frac{\pi}{2}$"]
    #     ax.set_yticks(ax_yticks*np.pi)
    #     ax.set_yticklabels(ax_yticklabels)

    for ax in (ax00, ax2):
        ax_yticks = np.arange(-1.0, 1.05, 0.5)
        ax_yticklabels = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$",
                        r"$+\frac{\pi}{2}$", r"$\pi$"]
        ax.set_yticks(ax_yticks*np.pi)
        ax.set_yticklabels(ax_yticklabels)

    ax00.set_ylim(-np.pi*1.05, np.pi*1.05)
    ax1.set_ylim(0, np.pi/2)
    ax2.set_ylim(-np.pi*1.05, np.pi*1.05)

    ax0.set_ylabel("Transmission intensity")
    ax00.set_ylabel("Transmission-matrix \n phase")
    ax1.set_ylabel(r"$\arctan(|c_1/c_2|)$")
    ax2.set_xlabel(r"Frequency $\nu$ (GHz)")
    ax2.set_ylabel(r"$\operatorname{Arg}(c_1/c_2)$")

    for ax in fig.get_axes():
        ax.set_xlim(7, 8.8)
        ax.axvline(x=7.8, color="k")

    if plot:
        plt.show()


if __name__ == '__main__':
    argh.dispatch_command(get_eigenvalues)
