#!/usr/bin/env python2.7

import numpy as np
from numpy import pi

class Transmission_Matrix(object):
    """Transmission matrix class."""
    
    def __init__(self, N=1.01, eta=0.1, delta=0.0, eps=None, eps_factor=1.0, L=50,
                 infile="S_matrix.dat"):
        """Calculate the length dependent transmission matrix based on the evolution
        operator obtained via an effective two-level system.
        
            Parameters:
            -----------
                L: float
                    Waveguide length
                eta: float
                    Dissipation coefficient
                delta: float
                    Constant to set y_EP (or, equivalently, y_EP -> y_EP + delta)
                eps: float
                    Set value for x_EP to eps (only done if not None)
                eps_factor: float
                    Constant to shift x_EP -> x_EP * eps_factor
                N: float
                    Number of open modes int(k*d/pi)
                infile: str
                    Input file which contains the S-matrix.
        """
        
        self.k0, self.k1 = [ np.sqrt(N**2 - n**2)*pi for n in (0,1) ]
        self.kr = self.k0 - self.k1
        
        self.eta = eta
        self.eps = eta/(2*np.sqrt(2*self.k0*self.k1))
        if eps:
            self.eps = eps
        else:
            self.eps *= eps_factor
    
        self.N = N
        self.delta = delta
        self.L = L
        self.x = np.arange(0,L,0.1)
        self.beta = np.sqrt(self.k1/(2*self.k0))
        self.B = self.kr * np.sqrt(self.k0/(2*self.k1))
        
        self.H11 = -self.k0 - 1j*self.eta/2.
        self.H22 = -self.k0 - self.delta - 1j*self.eta/2.*self.k0/self.k1
        self.H12 = self.B * (-1j) * self.eps
        self.H21 = self.B * (+1j) * self.eps
        self.D = np.sqrt((self.H11-self.H22)**2 + 4*self.H12*self.H21)
        self.lam1 = 0.5*(self.H11 + self.H22) + 0.5*self.D
        self.lam2 = 0.5*(self.H11 + self.H22) - 0.5*self.D
        self.gam1 = self.lam1 - self.H11
        self.gam2 = self.lam2 - self.H11
        
        self.infile = infile
        
    def get_U_at_EP(self):
        """Return the elements of the evolution operator U(x) at the
        exceptional point (EP).
        
            Returns:
            --------
                x: float
                U: (2,2) ndarray
                    Evolution operator.
        """
        exp = np.exp(-1j*self.x*0.5*(self.H11+self.H22))
        
        M11 = 1. + np.sqrt(self.H12*self.H21)*self.x
        M12 = -1j*self.H12*self.x
        M21 = -1j*self.H21*self.x
        M22 = 1. - np.sqrt(self.H12*self.H21)*self.x
        
        M11, M12, M21, M22 = [ exp*m for m in (M11,M12,M21,M22) ]
        
        #return self.x, M11, M12*self.beta, M21/self.beta, M22
        return self.x, M11, M12, M21, M22
    
    def get_U(self):
        """Return the elements of the evolution operator U(x) away from the
        exceptional point (EP).

            Returns:
            --------
                x: float
                U: (2,2) ndarray
                    Evolution operator.
        """
        
        pre = np.exp(-1j*self.lam1*self.x)/(self.D)
        exp = np.exp(1j*self.D*self.x)
        M11 = self.gam1*exp - self.gam2
        M12 = (1 - exp)*self.H12
        M21 = (1 - exp)*self.H21
        M22 = self.gam1 - self.gam2*exp
        
        M11, M12, M21, M22 = [ pre*m for m in (M11,M12,M21,M22) ]

        #return self.x, M11, M12*self.beta, M21/self.beta, M22
        return self.x, M11, M12, M21, M22
   
    def S_matrix(self):
        """Extract and return the elements of the scattering matrix S.
        
            Returns:
            --------
                x: float
                S: (2,2) ndarray
        """
        
        return np.loadtxt(self.infile, unpack=True,
                          dtype=complex, usecols=(0,5,6,7,8))
    

def compare_H_eff_to_S_matrix(N=1.01, eta=0.1, delta=0.0, eps=None,
                              eps_factor=1.0, L=50, infile="S_matrix.dat",
                              write_cfg=None):
    """Plot the predictions for the length dependent evolution operator U and
    compare them to their respective S-matrix elements.
    """
    
    import brewer2mpl
    import json
    import matplotlib.pyplot as plt
    
    if write_cfg:
        with open(write_cfg, "w") as f:
            data = json.dumps((N, eta, delta, eps, L, eps_factor))
            f.write(data)
    
    bmap = brewer2mpl.get_map('Paired', 'qualitative', 12)
    colors = bmap.mpl_colors
    
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    
    T = Transmission_Matrix(N, eta, delta, eps, eps_factor, L, infile)
   
    # time evolution operator U (effective model)
    x, U00, U01, U10, U11 = T.get_U()
    U = (U00, U01, U10, U11)
    labels = (r"$|U_{00}|$", r"$|U_{01}|$", r"$|U_{10}|$", r"$|U_{11}|$")
    
    for (i, u) in enumerate(U):
        #ax.semilogy(x, abs(u), c=colors[2*i], lw=2.0, label=labels[i])
        ax.plot(x, abs(u)**2, c=colors[2*i], lw=2.0, label=labels[i])

    # S-matrix (full system data)
    x, t00, t01, t10, t11 = T.S_matrix()
    trans = (t00, t01, t10, t11)
    labels = (r"$|t_{00}|$", r"$|t_{01}|$", r"$|t_{10}|$", r"$|t_{11}|$")
    
    for (i, t) in enumerate(trans):
        #ax.semilogy(abs(x), abs(t), color=colors[2*i+1],
        ax.plot(abs(x), abs(t)**2, color=colors[2*i+1],
                    lw=2.0, ls="-", marker="o", mew=0.35, ms=7.5, label=labels[i])

    ax.set_xlabel(r"$x$")
    ax.set_ylim(abs(t00).min(), abs(t00).max())

    if eps:
        ax.set_title((r"$N={N}$, $\eta={eta}$, "
                      r"$\delta={delta}$, " 
                      r"$\epsilon={eps}$").format(N=N, eta=eta, 
                                                  delta=delta, eps=eps))
    else:
        ax.set_title((r"$N={N}$, $\eta={eta}$, "
                      r"$\delta={delta}$, " 
                      r"$\epsilon={eps_factor} $"
                      r"$\cdot \epsilon_{{EP}}$").format(N=N, eta=eta, 
                                                         delta=delta, 
                                                         eps_factor=eps_factor))

    ax.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)      
    plt.show(block=False)
    
    
if __name__ == '__main__':
    import argh
    compare_H_eff_to_S_matrix.__doc__ = Transmission_Matrix.__init__.__doc__
    argh.dispatch_command(compare_H_eff_to_S_matrix)
