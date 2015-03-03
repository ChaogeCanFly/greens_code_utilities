#!/usr/bin/env python2.7

import numpy as np
from matplotlib import pyplot as plt

import argh


def main(mode='pic.complex_potential.0001.streu.0009.layer0.jpg',
         pot='pic.imag.potential.0008.layer0.jpg'):

    mode2 = plt.imread(mode)
    potential = plt.imread(pot)

    f = plt.figure(figsize=(100,1))
    plt.imshow(mode2)
    plt.imshow(potential, alpha=0.5)
    plt.axis("off")
    plt.savefig("overlay.jpg", bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    argh.dispatch_command(main)
