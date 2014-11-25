#!/usr/bin/env python2.7
"""Download and compile the following packages required to run greens_code:
    - cjpeg
    - expat
    - LARPACK *
    - PARPACK *
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

# TODO:
    - command-line interface
    - interactive selection of compilers
"""
import glob
import os
import socket
import shutil
import subprocess


SCC = 'g++'
SFC = 'gfortran'
MKL_DIR = "/home/doppler/intel/composerxe/"

PACKAGES = {'CJPEG':     'https://github.com/LuaDist/libjpeg.git',
            'EXPAT':     'https://github.com/cgwalters/expat-git-mirror.git',
            'ARPACK-NG': 'https://github.com/opencollab/arpack-ng.git',
            'PETSC':     'https://bitbucket.org/petsc/petsc.git'}

CONFIGURE_PETSC = ['--CXX=' + SCC,
                   '--FC=' + SFC,
                   '--download-metis',
                   '--download-mpich',
                   '--download-mumps',
                   '--download-parmetis',
                   '--download-scalapack',
                   '--download-sowing',
                   '--download-superlu',
                   '--with-clean=1']

INSTALL_DIR = os.path.join(os.getcwd(), 'dependencies')
os.mkdir(INSTALL_DIR)
os.chdir(INSTALL_DIR)

for name, url in PACKAGES.iteritems():
    print
    print 100*'#'
    print 'Obtaining {name}'.format(name=name)
    print

    for item in url.split("/"):

        if '.git' in item:
            cmd = "git clone {url}".format(url=url)
            subprocess.call(cmd.split())

            if 'petsc' in item:
                DIR = item.replace(".git", "")
                os.chdir(os.path.join(os.getcwd(), DIR))
                cmd = "./configure " + " ".join(CONFIGURE_PETSC)
                subprocess.call(cmd.split())
                shutil.rmtree(os.path.join(os.getcwd(), 'src'))

                PETSC_DIR = glob.glob("arch-*-debug")[0]
                PETSC_DIR = os.path.join(os.getcwd(), PETSC_DIR)
                shutil.rmtree(os.path.join(PETSC_DIR, 'externalpackages'))

            elif 'arpack-ng' in item:
                DIR = item.replace(".git", "")
                ARPACK_DIR = os.path.join(os.getcwd(), DIR)
                os.chdir(ARPACK_DIR)
                os.environ['LDFLAGS'] = "-L" + os.path.join(PETSC_DIR, "lib")
                cmd = "./configure --enable-mpi && make"
                subprocess.call(cmd, shell=True, env=os.environ)

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
                cmd = "autoreconf -f -i && ./configure --enable-shared && make"
                subprocess.call(cmd, shell=True)

        os.chdir(INSTALL_DIR)

# write entry for machines.inc
makeparams = {'HOSTNAME': socket.gethostname(),
              'SCC': SCC,
              'SFC': SFC,
              'CJPEG_DIR': CJPEG_DIR,
              'EXPAT_DIR': EXPAT_DIR,
              'PETSC_DIR': PETSC_DIR,
              'ARPACK_DIR': ARPACK_DIR}
makeparams.update({'MKL_DIR': MKL_DIR})

MAKE = """
  ifeq ($(MACHINE),{HOSTNAME})
    EXPAT = {EXPAT_DIR}/.libs/libexpat.a

    # C/C++
    SCC = {SCC}
    PCC = {PETSC_DIR}/bin/mpic++
    CFLAGS += -g -Wall -std=c++0x -O3
    CFLAGS += -DPARALLEL -DFROMGFORTRAN -DMKL_ILP64
    CFLAGS += -I{PETSC_DIR}/include -DUSE_BOOST
    CFLAGS += -I{MKL_DIR}/include
    CFLAGS += -I{EXPAT_DIR}/lib

    # FORTRAN
    SFC = {SFC}
    PFC = {PETSC_DIR}/bin/mpif90
    FFLAGS += -g -ffixed-line-length-none
    MODULESWITCH = GFORTRAN

    C_SUPER_LU = -I{PETSC_DIR}/include
    L_SUPER_LU = -L{PETSC_DIR}/lib

    C_MUMPS = -I{PETSC_DIR}/include
    C_MUMPS += -DUSE_MUMPS
    L_MUMPS = -L{PETSC_DIR}/lib
    L_MUMPS += -lzmumps -lmumps_common
    L_MUMPS += -lpord -lmetis -lparmetis -lscalapack -lmpifort

    PARPACK = -L{ARPACK_DIR}/PARPACK/.libs -lparpack
    LARPACK = -L{ARPACK_DIR}/.libs -larpack

    # MKL
    LFLAGS += -L{MKL_DIR}/lib/intel64
    MKL_FLAGS += -lmpi_f90 -l{SFC} -lmkl_gf_lp64
    MKL_FLAGS += -lmkl_sequential -lmkl_core -lpthread
    P_MKL_FLAGS += -lmkl_blacs_intelmpi_lp64
  endif
""".format(**makeparams)
print
print "Append the following to your machines.inc:"
print MAKE
print

# write entry for .bashrc
BASHRC = """
    # cjpeg executable
    export PATH=$PATH:{CJPEG_DIR}
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{MKL_DIR}/lib/intel64
""".format(**makeparams)
print
print "Add the following to your .bashrc:"
print BASHRC
print
