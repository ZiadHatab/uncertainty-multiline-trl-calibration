'''
@author: Ziad (zi.hatab@gmail.com)

This is an implementation of a co-planar waveguide model based on the references [1-3] (see the class CPW). 
Reference [1] is the base implementation, whereas [2] and [3] add modification on [1] to account for radiation effects.

[1] W. Heinrich, "Quasi-TEM description of MMIC coplanar lines including conductor-loss effects," 
in IEEE Transactions on Microwave Theory and Techniques, vol. 41, no. 1, pp. 45-52, Jan. 1993, doi: 10.1109/22.210228.
https://ieeexplore.ieee.org/document/210228

[2] F. Schnieder, T. Tischler and W. Heinrich, "Modeling dispersion and radiation characteristics of conductor-backed CPW with finite ground width," 
in IEEE Transactions on Microwave Theory and Techniques, vol. 51, no. 1, pp. 137-143, Jan. 2003, doi: 10.1109/TMTT.2002.806926.
https://ieeexplore.ieee.org/document/1159676 

[3] G. N. Phung, U. Arz, K. Kuhlmann, R. Doerner and W. Heinrich, "Improved Modeling of Radiation Effects in Coplanar Waveguides with Finite Ground Width," 
2020 50th European Microwave Conference (EuMC), 2021, pp. 404-407, doi: 10.23919/EuMC48046.2021.9338133.
https://ieeexplore.ieee.org/document/9338133
'''

# pip install numpy matplotlib scipy -U
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ssp
import scipy.optimize as so
import scipy.integrate as si

## Functions below are needed to implement the cpw model based on [1] and [2].
def K(x):
    # Complete elliptic integral of the first kind
    # Based on Eq. (14) in [1].
    return ssp.ellipk(x)

def Kp(k):
    # Based on Eq. (14) in [1].
    return K(np.sqrt(1-k**2))

def E(x):
    # Complete elliptic integral of the second kind
    # Based on Eq. (14) in [1].
    return ssp.ellipe(x)
 
def Ep(k):
    # Based on Eq. (14) in [1].
    return E(np.sqrt(1-k**2))

def PI(n,m):
    # Complete elliptic integral of the third kind
    # Based on Eq. (16) in [2].
    '''
    At the time of writing, scipy didn't support the Complete elliptic integral of the third kind.
    Therefore, I implemented it myself based on the paper below.
    Toshio Fukushima,
    Fast computation of a general complete elliptic integral of third kind by half and double argument transformations,
    Journal of Computational and Applied Mathematics,
    Volume 253, 2013, Pages 142-157, ISSN 0377-0427,
    https://doi.org/10.1016/j.cam.2013.04.015.
    '''
    B = lambda phi: np.cos(phi)**2/np.sqrt(1-m*np.sin(phi)**2)
    D = lambda phi: np.sin(phi)**2/np.sqrt(1-m*np.sin(phi)**2)
    J = lambda phi: np.sin(phi)**2/np.sqrt(1-m*np.sin(phi)**2)/(1-n*np.sin(phi)**2)
    return si.quad(B, 0, np.pi/2)[0] + si.quad(D, 0, np.pi/2)[0] + n*si.quad(J, 0, np.pi/2)[0]

def F(w,s,wg,t):
    # Based on Eq. (1) in [1].
    k0 = w/(w+2*s)
    k1 = k0*np.sqrt( ( 1-((w+2*s)/(w+2*s+2*wg))**2 )/( 1-(w/(w+2*s+2*wg))**2 ) )
    a = w/2
    b = w/2 + s
    pc0 = b/2/a/Kp(k0)**2
    pc1 = 1 + np.log(8*np.pi*a/(a+b)) + a/(a+b)*np.log(b/a)
    pc2 = pc1 - 2*a/b*Kp(k0)**2

    if t <= s/2:
        return K(k1)/Kp(k1) + pc0*( t/s*(pc1-np.log(2*t/s)) + (t/s)**2*(1-3/2*pc2+pc2*np.log(2*t/s)) )
    else:
        return K(k1)/Kp(k1) + pc0/8*(pc2 + 2) + t/s

## RLCG parameters based on [1].
def C_v1(w,s,wg,t,er):
    # This function only process a single frequency point.
    # er is the real part of the relative permittivity.
    # Based on Eq. (2) in [1].
    er  = er.real  # force to take the real-part
    ep0 = 8.8541878128e-12
    Fup  = F(w,s,wg,t)
    k0 = w/(w+2*s)
    k1 = k0*np.sqrt( (1-((w+2*s)/(w+2*s+2*wg))**2)/(1-(w/(w+2*s+2*wg))**2) )
    Flow = K(k1)/Kp(k1)
    return 2*ep0*(Fup + er*Flow)

def G_v1(w,s,wg,t,er,tand,f):
    # This function only process a single frequency point.
    # er is the real part of the relative permittivity.
    # Based on Eq. (2) in [1].
    er  = er.real  # force to take the real-part
    ep0 = 8.8541878128e-12
    k0 = w/(w+2*s)
    k1 = k0*np.sqrt( (1-((w+2*s)/(w+2*s+2*wg))**2)/(1-(w/(w+2*s+2*wg))**2) )
    Flow = K(k1)/Kp(k1)
    omega = 2*np.pi*f
    return 2*omega*ep0*er*tand*Flow

def R_v1(w,s,wg,t,f,sigma=5.8e7):
    # This function only process a single frequency point.
    # sigma is the conductivity. in [1] the letter kappa was used.
    # Based on Eqs. (6) and (7) in [1].
    mu0 = 1.25663706212e-6
    omega = 2*np.pi*f
    omega_c1 = np.sqrt(2)*4/mu0/sigma/t/w
    omega_c2 = 8/mu0/sigma*((w+t)/w/t)**2
    omega_g1 = 2/mu0/sigma/t/wg
    omega_g2 = 2/mu0/sigma*((2*wg+t)/wg/t)**2
    
    k0 = w/(w+2*s)
    a = w/2
    b = w/2 + s
    tH = t/2
    pc0 = b/2/a/Kp(k0)**2
    pc1 = 1 + np.log(8*np.pi*a/(a+b)) + a/(a+b)*np.log(b/a)
    pc2 = pc1 - 2*a/b*Kp(k0)**2
    pc3 = 2*b**2/a/(b+a)*Ep(k0)/Kp(k0)
    pc4 = (b-a)/(b+a)*(np.log(8*np.pi*a/(a+b))+a/b)
    pc5 = (b-a)/(b+a)*np.log(3)
    pc6 = (b-a)/(b+a)*np.log(24*np.pi*b*(a+b)/(b-a)**2)-b/(b+a)*np.log(b/a)
    
    if tH <= s/2:
        FcL = pc0/s*( 1/(a+b)*( np.pi*b + b*np.log(8*np.pi*a/(a+b)) - (b-a)*np.log((b-a)/(b+a)) - b*np.log(2*tH/s) ) \
                     + tH/s*( pc1*pc3 - pc2 - b/a*pc4 + pc5 + (pc2-pc3+b/a-1-pc5)*np.log(2*tH/s) ) \
                         + (tH/s)**2*( pc3*(1-3/2*pc1) + 3/2*pc1 - 2*pc2 + 1 + 3/2*b/a*pc4 - b/a*(b-a)/(b+a) + (2*pc2+pc1*(pc3-1)-b/a*pc4)*np.log(2*tH/s) ) \
                             )
        FgL = pc0/s*( 1/(a+b)*( np.pi*a + a*np.log(8*np.pi*b/(b-a)) + b*np.log((b-a)/(b+a)) - a*np.log(2*tH/s) ) \
                     + tH/s*( a/b*pc1*pc3 + (1-a/b)*pc1 - pc2 - pc4 - pc5 + (-a/b*pc3+pc2+a/b-1+pc5)*np.log(2*tH/s) ) \
                         + (tH/s)**2*( a/b*pc3*(1-3/2*pc1) + 3/2*a/b*pc1 - 2*pc2 + 2 - a/b + 3/2*pc4 - (b-a)/(b+a) + (2*pc2+a/b*pc1*(pc3-1)-pc4)*np.log(2*tH/s) ) \
                             )
    else:
        FcL = 1/2/s + tH/s**2 + pc0/s*( np.pi*b/(a+b) + pc6/2 + 1/8*( -pc1 + pc3*(pc1+2) - b/a*pc4 - 2*(a**2+b**2)/a/(a+b) ) )
        FgL = 1/2/s + tH/s**2 + pc0/s*( np.pi*a/(a+b) + pc6/2 + 1/8*( -a/b*pc1 + a/b*pc3*(pc1+2) - pc4 - 2*(a**2+b**2)/b/(a+b) ) )
            
    F0 = F(w,s,wg,t/2)
    
    Rc0 = 1/sigma/w/t
    Rc1 = np.sqrt(omega_c2*mu0/sigma/2)*FcL/4/F0**2
    Rg0 = 1/2/sigma/wg/t
    Rg1 = np.sqrt(omega_g2*mu0/sigma/2)*FgL/4/F0**2
    
    vc = np.log(Rc0/Rc1)/np.log(omega_c1/omega_c2)
    vg = np.log(Rg0/Rg1)/np.log(omega_g1/omega_g2)
    
    gamma_c = (omega_c1/omega_c2)**2 # (w*t/np.sqrt(2)/(w+t)**2)**2
    gamma_g = (omega_g1/omega_g2)**2 # (wg*t/(2*wg+t)**2)**2
    
    a4 = lambda v, gamma: ( gamma*v + 1/4*(1/2-v)*(4-v*(1-gamma**2)) )/( 4-v-1/4*(1/2-v)*(4-v*(1-gamma**2)) )
    a3 = lambda v, gamma: 1/4*(1/2-v)*(1 + a4(v,gamma))
    a2 = lambda v, gamma: 1/gamma*(a4(v,gamma) - a3(v,gamma))
    a1 = lambda v, gamma: a2(v,gamma) + gamma*a3(v,gamma)
    
    ac1,ac2,ac3,ac4 = a1(vc,gamma_c),a2(vc,gamma_c),a3(vc,gamma_c),a4(vc,gamma_c)
    ag1,ag2,ag3,ag4 = a1(vg,gamma_g),a2(vg,gamma_g),a3(vg,gamma_g),a4(vg,gamma_g)
    
    if omega <= omega_c1:
        Rc = Rc0*(1 + ac1*(omega/omega_c1)**2)
    elif omega_c1 < omega <= omega_c2:
        Rc = Rc1*(omega/omega_c2)**vc*( 1 + ac2*(omega_c1/omega)**2 + ac3*(omega/omega_c2)**2 )
    else:
        Rc = np.sqrt(omega*mu0/2/sigma)*FcL/4/F0**2*( 1 + ac4*(omega_c2/omega)**2 )
    
    if omega <= omega_g1:
        Rg = Rg0*(1 + ag1*(omega/omega_g1)**2)
    elif omega_g1 < omega <= omega_g2:
        Rg = Rg1*(omega/omega_g2)**vg*( 1 + ag2*(omega_g1/omega)**2 + ag3*(omega/omega_g2)**2 )
    else:
        Rg = np.sqrt(omega*mu0/2/sigma)*FgL/4/F0**2*( 1 + ag4*(omega_g2/omega)**2 )
    
    return Rc + Rg


def L_v1(w,s,wg,t,f,sigma=5.8e7):
    # This function only process a single frequency point.
    # sigma is the conductivity. in [1] the letter kappa was used.
    # Based on Eq. (9) in [1]
    mu0 = 1.25663706212e-6
    omega = 2*np.pi*f
    omega_L0 = 4/mu0/sigma/t/wg
    omega_L1 = 4/mu0/sigma/t/w
    omega_L2 = 18/mu0/sigma/t**2
    
    gL  = lambda x: (1/12*t**2-1/2*x**2)*np.log(1+(x/t)**2) + 1/12*x**4/t**2*np.log(1+(t/x)**2) - 2/3*x*t*( np.arctan(x/t)+(x/t)**2*np.arctan(t/x) ) 
    LDC = lambda w1,w2: mu0/8/np.pi*( 4/w1**2*gL(w1) \
                                     + 1/w2**2*( gL(w1+2*s) + gL(w1+2*w2+2*s) + 2*gL(w2) - 2*gL(w1+w2+2*s) ) \
                                         - 4/w1/w2*( gL(w1+w2+s) - gL(w1+s) + gL(s) - gL(w2+s) ) )
    
    k0 = w/(w+2*s)
    k1 = k0*np.sqrt( (1-((w+2*s)/(w+2*s+2*wg))**2)/(1-(w/(w+2*s+2*wg))**2) )
    k2 = k0*np.sqrt( (1-((w+2*s)/(4*w+2*s))**2)/(1-(w/(4*w+2*s))**2) )
    a = w/2
    b = w/2 + s
    tH = t/2
    pc0 = b/2/a/Kp(k0)**2
    pc1 = 1 + np.log(8*np.pi*a/(a+b)) + a/(a+b)*np.log(b/a)
    pc2 = pc1 - 2*a/b*Kp(k0)**2
    pc3 = 2*b**2/a/(b+a)*Ep(k0)/Kp(k0)
    pc4 = (b-a)/(b+a)*(np.log(8*np.pi*a/(a+b))+a/b)
    pc5 = (b-a)/(b+a)*np.log(3)
    pc6 = (b-a)/(b+a)*np.log(24*np.pi*b*(a+b)/(b-a)**2)-b/(b+a)*np.log(b/a)
    
    if tH <= s/2:
        FcL = pc0/s*( 1/(a+b)*( np.pi*b + b*np.log(8*np.pi*a/(a+b)) - (b-a)*np.log((b-a)/(b+a)) - b*np.log(2*tH/s) ) \
                     + tH/s*( pc1*pc3 - pc2 - b/a*pc4 + pc5 + (pc2-pc3+b/a-1-pc5)*np.log(2*tH/s) ) \
                         + (tH/s)**2*( pc3*(1-3/2*pc1) + 3/2*pc1 - 2*pc2 + 1 + 3/2*b/a*pc4 - b/a*(b-a)/(b+a) + (2*pc2+pc1*(pc3-1)-b/a*pc4)*np.log(2*tH/s) ) \
                             )
        FgL = pc0/s*( 1/(a+b)*( np.pi*a + a*np.log(8*np.pi*b/(b-a)) + b*np.log((b-a)/(b+a)) - a*np.log(2*tH/s) ) \
                     + tH/s*( a/b*pc1*pc3 + (1-a/b)*pc1 - pc2 - pc4 - pc5 + (-a/b*pc3+pc2+a/b-1+pc5)*np.log(2*tH/s) ) \
                         + (tH/s)**2*( a/b*pc3*(1-3/2*pc1) + 3/2*a/b*pc1 - 2*pc2 + 2 - a/b + 3/2*pc4 - (b-a)/(b+a) + (2*pc2+a/b*pc1*(pc3-1)-pc4)*np.log(2*tH/s) ) \
                             )
    else:
        FcL = 1/2/s + tH/s**2 + pc0/s*( np.pi*b/(a+b) + pc6/2 + 1/8*( -pc1 + pc3*(pc1+2) - b/a*pc4 - 2*(a**2+b**2)/a/(a+b) ) )
        FgL = 1/2/s + tH/s**2 + pc0/s*( np.pi*a/(a+b) + pc6/2 + 1/8*( -a/b*pc1 + a/b*pc3*(pc1+2) - pc4 - 2*(a**2+b**2)/b/(a+b) ) )
    
    F0 = F(w,s,wg,t/2)
    F1 = F0 + K(k2)/Kp(k2) - K(k1)/Kp(k1)
    
    Linf = mu0/4/F0
    LDC0 = LDC(w,wg)
    Lz1  = LDC(w,3/2*w) - mu0/4/F1
    Lz2  = np.sqrt( mu0/2/omega_L2/sigma )*(FcL + FgL)/4/F0**2
    
    vz1 = np.log((LDC0 - Linf)/Lz1)/np.log(omega_L0/omega_L1)
    vz2 = np.log(Lz1/Lz2)/np.log(omega_L1/omega_L2)
    
    n1 = (w/wg)**4*vz1/(4-vz1)
    n2 = (w/wg)**2*vz1/(4-vz1)
    n3 = (2*t/9/w)**3*(vz2-1/2)/(vz2+5/2)
    n4 = (2*t/9/w)*(vz2+1/2)/(vz2+5/2)
    
    aL3 = ( (vz2 - vz1)*(1 + n1)*(1 - n4) * 4*n2 + n4*(1 - 3*n1) )/( (vz1 - vz2)*(1 + n1)*(1 - n3) + 4 - n3*(1 - 3*n1) )
    aL2 = 1/(1 + n1)*( aL3*(1 - n3) - n2 - n4 )
    aL4 = -9/2*w/t*( n4 + aL3*n3 )
    aL5 = (2/9*t/w)**2*aL3 + aL4
    aL1 = vz1/(4 - vz1) + n2*aL2
    aL0 = (1 - Linf/LDC0)*( aL1 + (w/wg)**2*aL2 )
    
    if omega <= omega_L0:
        return LDC0*( 1 + aL0*(omega/omega_L0)**2 )
    elif omega_L0 < omega <= omega_L1:
        return Linf + Lz1*(omega/omega_L1)**vz1*( 1 + aL1*(omega_L0/omega)**2 + aL2*(omega/omega_L1)**2 )
    elif omega_L1 < omega <= omega_L2:
        return Linf + Lz2*(omega/omega_L2)**vz2*( 1 + aL3*(omega_L1/omega)**2 + aL4*(omega/omega_L2) )
    else:
        return Linf + np.sqrt(mu0/2/omega/sigma)*(FcL + FgL)/4/F0**2*(1 + aL5*(omega_L2/omega))


def R_rad(w,s,wg,t,f,er):
    # Implementation of added radiation loss as series per-unit-length resistance. 
    # Based on Eq. (16) in [2].
    mu0 = 1.25663706212e-6
    ep0 = 8.8541878128e-12
    omega = 2*np.pi*f
    er  = er.real
    erq = (er + 1)/2 
    a = w/2
    b = w/2 + s
    c = w/2 + s + wg
    r1 = a/b
    r2 = np.sqrt((c**2-b**2)/(c**2-a**2))
    r = r1*r2
    return mu0**3*ep0**2*omega**5/16/er*(er-erq)**3*(np.sqrt(8)-2.75)*( (b**2-a**2)*(K(r)-PI(r1,r)-PI(r2,r))/K(r) )**2


def get_all_paras(x, cpw):
    '''
    This function is used to collect all output parameters as real-valued equivalent to compute 
    the Jacobian of them with respect to input parameters.
    '''
    # x = [ w,s,wg,t,f,Dk,Df,sigma ]  # this is the order of the inputs
    cpw.w     = x[0]
    cpw.s     = x[1]
    cpw.wg    = x[2]
    cpw.t     = x[3]
    cpw.er    = x[4]*(1-1j*x[5])
    cpw.sigma = x[6]
    cpw.update()
    # convert complex-valued array to real-valued equivalent
    hh = lambda x: np.kron(x.real, [1,0]) + np.kron(x.imag, [0,1])
    # cpw line parameters
    gamma = cpw.gamma
    h_gamma = hh(gamma)
    ereff = cpw.ereff
    h_ereff = hh(ereff)
    Z0 = cpw.Z0
    h_Z0 = hh(Z0)
    # RLGC parameters
    R = cpw.R
    h_R = hh(R)
    L = cpw.L
    h_L = hh(L)
    G = cpw.G
    h_G = hh(G)
    C = cpw.C
    h_C = hh(C)
    return np.hstack((h_gamma, h_ereff, h_Z0, h_R, h_L, h_G, h_C))

class CPW:
    """
    Analytical model of Co-planar waveguide (CPW) based on [1-3] (see comments at top of this file).

    Parameters
    ----------
    w : number
        width of the signal trace (center conductor).
    s : number
        spacing between the signal trace and ground plane (the same for both grounds)
    wg : number
        width of the ground plane (the same for both grounds)
    t : number
        thickness of the conductor (the same for signal and grounds)
    f : number or 1d-array
        frequency 
    er : complex number
        complex relative permittivity of the substrate. If you have the real-values Dk and Df, then er = Dk*(1-1j*Df)
    sigma : number (default 5.8e7)
        conductivity of the conductor (the same for signal and grounds)
    """

    def __init__(self,w,s,wg,t,f,er,sigma=5.8e7):
        self.w     = w
        self.s     = s
        self.wg    = wg
        self.t     = t
        self.f     = np.atleast_1d(f)
        self.er    = er
        self.sigma = sigma
        self.update()  # run the code

    def update(self):
        mu0 = 1.25663706212e-6
        ep0 = 8.8541878128e-12
        c0  = 1/np.sqrt(mu0*ep0) # 299792458 # speed of light in vacuum (m/s)
        
        w       = self.w
        s       = self.s
        wg      = self.wg
        t       = self.t
        f       = self.f
        er_real = self.er.real
        tand    = -self.er.imag/self.er.real
        sigma   = self.sigma
        
        omega = 2*np.pi*f
        
        # based on [1]
        C = np.array([C_v1(w,s,wg,t,er_real) for ff in f])
        G = np.array([G_v1(w,s,wg,t,er_real,tand,ff) for ff in f])
        R = np.array([R_v1(w,s,wg,t,ff,sigma) for ff in f])
        L = np.array([L_v1(w,s,wg,t,ff,sigma) for ff in f])

        # correction based on [2]
        # updating C
        wtot = w + 2*s + 2*wg
        fg = 2*c0/wtot/np.sqrt(2*(er_real-1))
        d = w + 2*s
        p = 2.86465*(d/wtot)**2/(0.15075+d/wtot)
        erq = (er_real + 1)/2
        x = 1 + (np.sqrt(er_real/erq)-1)*p*(f/fg)**2  # correction factor
        C_corr = (x-1)*(C - R*G/L/omega**2)
        C = C + C_corr
        # updating R
        R_corr = np.array([R_rad(w,s,wg,t,ff,er_real) for ff in f])
        R = R + R_corr        
        gamma = np.sqrt( (R + 1j*omega*L)*(G + 1j*omega*C) )
      
        # correction based on [3]
        # updating alpha = Re{gamma}
        f1 = 1 + 0.45*(w/d)**4
        f2 = 1.87 + 273.18/( 47.6 + 1.29*(er_real-9)**2 )
        f3 = wg/wtot
        fg1 = 2*c0/1.1/wtot/np.sqrt(2*(er_real-1))
        gamma = gamma.real*( 1 + f1*f2*f3/(1 + 19.83*(f/fg1-1)**2) ) + 1j*gamma.imag

        # Final results
        self.gamma = gamma
        self.ereff = -(c0/2/np.pi/f*self.gamma)**2
        self.Z0    = self.gamma/(G + 1j*omega*C)
        self.R     = (self.gamma*self.Z0).real
        self.L     = (self.gamma*self.Z0).imag/omega
        self.G     = (self.gamma/self.Z0).real
        self.C     = (self.gamma/self.Z0).imag/omega
        
        # set Jacobians to None if newly evaluated (you need to run update_jac() to compute them)
        self.jac_gamma = None
        self.jac_ereff = None
        self.jac_Z0    = None
        self.jac_Gamma = None
        self.jac_R     = None
        self.jac_L     = None
        self.jac_G     = None
        self.jac_C     = None
        
    def update_jac(self):
        '''
        Updates the jacobian of the parameters with respect to the input parameters.
            
        '''
        
        # these are the input parameters to which the Jacobian is computed with respect to (same order!).
        w       = self.w
        s       = self.s
        wg      = self.wg
        t       = self.t
        er_real = self.er.real
        tand    = -self.er.imag/self.er.real
        sigma   = self.sigma
        
        N = len(self.f)
        M = 2*N
        eps = np.sqrt(np.finfo(float).eps)
        x   = [w, s, wg, t, er_real, tand, sigma]  # the order of the input parameters for the Jacobian
        big_J = so.approx_fprime(x, get_all_paras, [eps]*(len(x)-1) + [eps**2], self)
        
        # split the jacobian for the corresponding output parameters
        self.jac_gamma = big_J[:M].reshape((N,2,-1))
        self.jac_ereff = big_J[M:2*M].reshape((N,2,-1))
        self.jac_Z0    = big_J[2*M:3*M].reshape((N,2,-1))
        self.jac_Gamma = np.array([ np.array([[(1/2/z).real, (1j/2/z).real],[(1/2/z).imag, (1j/2/z).imag]])@Jz for z,Jz in zip(self.Z0, self.jac_Z0) ])
        
        self.jac_R     = big_J[3*M:4*M:2].reshape((N,1,-1))
        self.jac_L     = big_J[4*M:5*M:2].reshape((N,1,-1))
        self.jac_G     = big_J[5*M:6*M:2].reshape((N,1,-1))
        self.jac_C     = big_J[6*M:7*M:2].reshape((N,1,-1))
        
        # undo the changes done by the function so.approx_fprime().
        self.w     = w
        self.s     = s
        self.wg    = wg
        self.t     = t
        self.er    = er_real*(1-1j*tand)
        self.sigma = sigma
        

if __name__=='__main__':
    # constants
    mu0 = 1.25663706212e-6
    ep0 = 8.8541878128e-12
    c0  = 1/np.sqrt(mu0*ep0) # 299792458   # speed of light in vacuum (m/s)
    # useful functions
    mag2db = lambda x: 20*np.log10(abs(x))
    db2mag = lambda x: 10**(x/20)
    gamma2ereff = lambda x,f: -(c0/2/np.pi/f*x)**2
    ereff2gamma = lambda x,f: 2*np.pi*f/c0*np.sqrt(-(x-1j*np.finfo(float).eps))  # eps to ensure positive square-root
    gamma2dbmm  = lambda x: mag2db(np.exp(x.real*1e-3))  # losses dB/mm
    
    w, s, wg, t = 46.5e-6, 26.3e-6, 271.6e-6, 4.9e-6
    f  = np.linspace(1, 150, 300)*1e9
    Dk = 9.9
    Df = 0.0
    sigma = 4.11e7 
    
    cpw = CPW(w,s,wg,t,f,Dk*(1-1j*Df),sigma)
    gamma_cpw = cpw.gamma
    Z0_cpw = cpw.Z0
    
    plt.figure()
    plt.plot(f*1e-9, gamma2dbmm(gamma_cpw), lw=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Loss (dB/mm)')
    
    plt.figure()
    plt.plot(f*1e-9, gamma2ereff(gamma_cpw,f).real, lw=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Relative effective permittivity')
    
    plt.figure()
    plt.plot(f*1e-9, Z0_cpw.real, lw=2, label='real')
    plt.plot(f*1e-9, Z0_cpw.imag, lw=2, label='imag')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Characteristic impedance (Ohm)')
    
    # example when you update values
    cpw.t = 10e-6
    cpw.update()
    Z0_cpw = cpw.Z0
    plt.figure()
    plt.plot(f*1e-9, Z0_cpw.real, lw=2, label='real')
    plt.plot(f*1e-9, Z0_cpw.imag, lw=2, label='imag')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Characteristic impedance (Ohm)')
    
    plt.show()

# EOF