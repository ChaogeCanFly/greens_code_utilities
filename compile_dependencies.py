#!/usr/bin/env python2.7
"""Download and compile the following packages required to run greens_code:
    - cjpeg
    - expat
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
"""
import glob
import os
import socket
import shutil
import subprocess
import tarfile
import urllib



PACKAGES = {'CJPEG':     'https://github.com/LuaDist/libjpeg.git',
            # 'EXPAT':   'http://sourceforge.net/projects/expat/files/expat/2.0.1/expat-2.0.1.tar.gz/download',
            'EXPAT':     'https://github.com/cgwalters/expat-git-mirror.git',
            'ARPACK-NG': 'https://github.com/opencollab/arpack-ng.git',
            'PETSC':     'https://bitbucket.org/petsc/petsc.git'}

SCC = 'g++'
SFC = 'gfortran'
CONFIGURE_PETSC = ['--download-mumps',
                   '--download-mpich',
                   '--download-boost',
                   '--download-sowing',
                   '--download-fftw',
                   '--download-superlu',
                   '--download-scalapack',
                   # '--download-cmake',
                   '--with-c++-support=1',
                   '--with-shared-libraries=1',
                   '--with-fortran-kernels=1']

INSTALL_DIR = os.path.join(os.getcwd(), 'dependencies')
os.mkdir(INSTALL_DIR)
os.chdir(INSTALL_DIR)

for name, url in PACKAGES.iteritems():
    print 100*'#'
    print 'Obtaining {name}'.format(name=name)
    print
    for item in url.split("/"):
        # if '.tar.gz' in item:
        #     urllib.urlretrieve(url, item)
        #     tar = tarfile.open(item)
        #     tar.extractall()
        #     tar.close()
        #
        #     if 'expat' in item:
        #         DIR = item.replace(".tar.gz", ""))
        #         EXPAT_DIR = os.path.join(os.getcwd(), DIR)
        #         os.chdir(EXPAT_DIR)
        #         cmd = "./configure && make"
        #         subprocess.call(cmd, shell=True)

        if '.git' in item:
            cmd = "git clone {url}".format(url=url)
            subprocess.call(cmd.split())

            if 'arpack-ng' in item:
                DIR = item.replace(".git", "")
                ARPACK_DIR = os.path.join(os.getcwd(), DIR)
                os.chdir(ARPACK_DIR)
                cmd = "./configure && make"
                subprocess.call(cmd, shell=True)

            elif 'libjpeg' in item:
                DIR = item.replace(".git", "")
                CJPEG_DIR = os.path.join(os.getcwd(), DIR)
                os.chdir(CJPEG_DIR)
                cmd = "./configure && make"
                subprocess.call(cmd, shell=True)

            elif 'expat' in item:
                DIR = item.replace(".git", "")
                EXPAT_DIR = os.path.join(os.getcwd(), DIR)
                os.chdir(EXPAT_DIR)
                cmd = "autoreconf --force --install && ./configure && make"
                subprocess.call(cmd, shell=True)

            elif 'petsc' in item:
                DIR = item.replace(".git", "")
                os.chdir(os.path.join(os.getcwd(), DIR))
                cmd = "./configure " + " ".join(CONFIGURE_PETSC)
                subprocess.call(cmd.split())
                shutil.rmtree(os.path.join(os.getcwd(), 'src'))

                PESC_DIR = glob.glob("arch-*-debug")[0]
                PESC_DIR = os.path.join(os.getcwd(), PESC_DIR)
                shutil.rmtree(os.path.join(PESC_DIR, 'externalpackages'))

        os.chdir(INSTALL_DIR)
    print


# write entry for machines.inc
makeparams = {'HOSTNAME': socket.gethostname(),
              'SCC': SCC,
              'SFC': SFC,
              'CJPEG_DIR': CJPEG_DIR,
              'EXPAT_DIR': EXPAT_DIR,
              'PESC_DIR': PESC_DIR,
              'ARPACK_DIR': ARPACK_DIR,
              'MKL_DIR': None}

MAKE = """
  ifeq ($(MACHINE),{HOSTNAME})
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

    C_MUMPS = -I{PESC_DIR}/include
    C_MUMPS += -DUSE_MUMPS
    P_MUMPS = -L{PESC_DIR}/lib
    P_MUMPS += -lzmumps -lmumps_common

    PARPACK = -lparpack
    LARPACK = -L{ARPACK_DIR}/lib -larpack

    # MKL
    LFLAGS += -L{MKL_DIR}/lib/intel64
    P_MKL_FLAGS += -lmkl_blacs_intelmpi_lp64
    MKL_FLAGS += -lmpi_f90 -l{SFC} -lmkl_gf_lp64
    MKL_FLAGS += -lmkl_sequential -lmklcore -lpthread
  endif
""".format(**makeparams)

BASHRC = """
    export PATH=$PATH:{CJPEG_DIR}
    export LD_LIBRARY_PATH={MKL_DIR}/lib/intel64
""".format(**makeparams)

print
print "Append the following to your machines.inc:"
print MAKE
print
print "Add the following to your .bashrc:"
print BASHRC
print
