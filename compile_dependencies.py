#!/usr/bin/env python2.7
"""Download and compile the following packages required to run greens_code:

    - EXPAT
    - LARPACK *
    - PARPACK *
    - BOOST **
    - FFTW3 **
    - MPICH **
    - MUMPS **
    - SUPER_LU **

    * via ARPACK-NG
        https://github.com/opencollab/arpack-ng.git
    ** via PETSc
        https://bitbucket.org/petsc/petsc.git

The following packages can only be obtained after (manual) registration:

    - Intel Math Kernel Library (MKL):
        https://software.intel.com/en-us/intel-education-offering

Additional requirements:
    - git
    - cmake
    - cjpeg
"""
import glob
import os
import socket
import shutil
import subprocess
import tarfile
import urllib


INSTALL_DIR = os.path.join(os.getcwd(), "libraries")

PACKAGES = {'EXPAT':   'http://sourceforge.net/projects/expat/files/expat/2.0.1/expat-2.0.1.tar.gz/download',
            'ARPACK-NG': 'https://github.com/opencollab/arpack-ng.git',
            'PETSC': 'https://bitbucket.org/petsc/petsc.git'}
# 'EXPAT':     'https://github.com/cgwalters/expat-git-mirror.git',

CONFIGURE_PETSC = ['--download-mumps',
                   '--download-mpich',
                   '--download-boost',
                   '--download-sowing',
                   '--download-fftw',
                   '--download-superlu',
                   '--download-scalapack',
                   '--download-metis',
                   '--download-parmetis',
                   '--with-c++-support=1',
                   '--with-shared-libraries=1',
                   '--with-fortran-kernels=1']
CONFIGURE_PETSC.append('PATSC-ARCH=gfortran')

os.mkdir(INSTALL_DIR)
os.chdir(INSTALL_DIR)

for name, url in PACKAGES.iteritems():
    print 100*'#'
    print 'Obtaining {name}'.format(name=name)
    print
    for item in url.split("/"):
        if '.tar.gz' in item:
            urllib.urlretrieve(url, item)
            tar = tarfile.open(item)
            tar.extractall()
            tar.close()

            if 'expat' in item:
                EXPAT_DIR = item.replace(".tar.gz", "")
                os.chdir(os.path.join(os.getcwd(), EXPAT_DIR))
                cmd = "./configure && make"
                subprocess.call(cmd, shell=True)

        if '.git' in item:
            cmd = "git clone {url}".format(url=url)
            subprocess.call(cmd.split())

            if 'arpack-ng' in item:
                DIR = item.replace(".git", "")
                os.chdir(os.path.join(os.getcwd(), DIR))
                cmd = "./configure && make"
                subprocess.call(cmd, shell=True)
                ARPACK_DIR = os.path.join(os.getcwd(), DIR)

            # elif 'expat' in item:
            #     DIR = item.replace("-git-mirror.git", "")
            #     cmd = "./configure && make"
            #     subprocess.call(cmd, shell=True)
            #     EXPAT_DIR = os.path.join(os.getcwd(), item.replace(".git", ""))

            elif 'petsc' in item:
                DIR = item.replace(".git", "")
                os.chdir(os.path.join(os.getcwd(), DIR))
                cmd = "./configure " + " ".join(CONFIGURE_PETSC)
                subprocess.call(cmd.split())
                shutil.rmtree(os.path.join(os.getcwd(), DIR, 'src'))

                PESC_DIR = glob.glob("arch-*-debug")
                PESC_DIR = os.path.join(os.getcwd(), DIR, PESC_DIR)
                shutil.rmtree(os.path.join(PESC_DIR, 'externalpackages'))

        os.chdir(INSTALL_DIR)
    print


# write entry for machines.inc
makeparams = {'hostname': socket.gethostname(),
              'SCC': 'g++',
              'SFC': 'gfortran',
              'PESC_DIR': PESC_DIR,
              'ARPACK_DIR': ARPACK_DIR,
              'MKL_DIR': None}

MAKE = """
    EXPAT = {EXPAT_DIR}/libs/libexpat.la

    # C/C++
    SCC = g++
    PCC = {PESC_DIR}/bin/mpic++
    CFLAGS += -g -Wall -std=c++0x
    CFLAGS += -DPARALLEL -DFROMGFORTRAN -DMKL_ILP64
    CFLAGS += -I{PESC_DIR}/include -DUSE_BOOST
    CFLAGS += -I{MKL_DIR}/include
    CFLAGS += -I{EXPAT_DIR}/lib -03

    # FORTRAN
    SFC = {SFC}
    PFC = {PESC_DIR}/bin/mpif90
    FFLAGS += -g -ffixed-line-length-none
    MODULESWITCH = GFORTRAN

    C_SUPER_LU = -I{PESC_DIR}/include
    L_SUPER_LU = -L{PESC_DIR}/lib

    C_MUMPS = -I{PESC_DIR)/include
    C_MUMPS += -DUSE_MUMPS
    P_MUMPS = -L{PESC_DIR)/lib
    P_MUMPS += -lzmumps -lmumps_common

    PARPACK = -lparpack
    LARPACK = -L{ARPACK_DIR}/lib -larpack

    LFLAGS += -L{MKL_DIR}/lib/intel64
    P_MKL_FLAGS += -lmkl_blacs_intelmpi_lp64
    MKL_FLAGS += -lmpi_f90 -l{SFC} -lmkl_gf_lp64
    MKL_FLAGS += -lmkl_sequential -lmklcore -lpthread
""".format(**makeparams)

print "Append the following to your machine.inc:"
print MAKE
