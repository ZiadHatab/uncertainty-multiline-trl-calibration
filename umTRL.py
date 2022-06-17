"""
@author: Ziad Hatab (zi.hatab@gmail.com)

This is an implementation of the multiline TRL calibration algorithm with linear 
uncertainty propagation capabilities. This is an extension of my original 
mTRL algorithm [1] to account for uncertainties with the help of 
METAS UncLib [2]. Additional math is included to handle various uncertainty types.

Developed at:
    Institute of Microwave and Photonic Engineering,
    Graz University of Technology (TU Graz), Austria

[1] Z. Hatab, M. Gadringer, and W. B¨osch, 
"Improving the reliability of the multiline trl calibration algorithm," 
in 98th ARFTG Microwave Measurement Conference, Las Vegas, NV, USA, 2022.

[2] M. Zeier, J. Hoffmann, and M. Wollensack, 
"Metas.UncLib—a measurement uncertainty calculator for advanced problems," 
Metrologia, vol. 49, no. 6, pp. 809–815, nov 2012.
"""

import numpy as np
import skrf as rf
import scipy.optimize as so   # optimization package (for initial gamma)
import metas_unclib as munc
munc.use_linprop()

c0 = 299792458   # speed of light in vacuum (m/s)

def metas_or_numpy_funcs(metas=False):
    # this is my way to switch between metas and numpy functions
    if metas:
        dot   = munc.ulinalg.dot
        inv   = munc.ulinalg.inv
        eig   = munc.ulinalg.eig
        solve = munc.ulinalg.solve
        conj  = munc.umath.conj
        exp   = munc.umath.exp
        log   = munc.umath.log
        acosh = munc.umath.acosh
        sqrt  = munc.umath.sqrt
        get_value = munc.get_value
        ucomplex  = munc.ucomplex
    else:
        dot   = np.dot
        inv   = np.linalg.inv
        eig   = np.linalg.eig
        solve = np.linalg.solve
        conj  = np.conj
        exp   = np.exp
        log   = np.log
        acosh = np.arccosh
        sqrt  = np.sqrt
        get_value = lambda x: x
        ucomplex  = complex
    return dot, inv, eig, solve, conj, exp, log, acosh, sqrt, get_value, ucomplex

def correct_switch_term(S, G21, G12):
    # correct switch terms of measured S-parameters at a single frequency point
    # G21: forward (sourced by port-1)
    # G12: reverse (sourced by port-2)
    S_new = S.copy()
    S_new[0,0] = (S[0,0]-S[0,1]*S[1,0]*G21)/(1-S[0,1]*S[1,0]*G21*G12)
    S_new[0,1] = (S[0,1]-S[0,0]*S[0,1]*G12)/(1-S[0,1]*S[1,0]*G21*G12)
    S_new[1,0] = (S[1,0]-S[1,1]*S[1,0]*G21)/(1-S[0,1]*S[1,0]*G21*G12)
    S_new[1,1] = (S[1,1]-S[0,1]*S[1,0]*G12)/(1-S[0,1]*S[1,0]*G21*G12)
    return S_new

def S2T(S):
    # convert S- to T-parameters at a single frequency point
    T = S.copy()
    T[0,0] = -(S[0,0]*S[1,1]-S[0,1]*S[1,0])
    T[0,1] = S[0,0]
    T[1,0] = -S[1,1]
    T[1,1] = 1
    return T/S[1,0]

def T2S(T):
    # convert T- to S-parameters at a single frequency point
    S = T.copy()
    S[0,0] = T[0,1]
    S[0,1] = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    S[1,0] = 1
    S[1,1] = -T[1,0]
    
    return S/T[1,1]

def findereff(x, *argv):
    # objective function to estimate ereff (details in [2])
    meas = argv[0]
    line_lengths = argv[1]
    w = 2*np.pi*argv[2]
    ereff = x[0] + 1j*x[1]
    gamma = w/c0*np.sqrt(-ereff)
    ex = np.exp(gamma*line_lengths)
    E = np.outer(ex, 1/ex)
    model = E + E.T
    error = meas - model
    return (error*error.conj()).real.sum() # np.linalg.norm(error, ord='fro')**2
    #return np.linalg.norm(error, ord='nuc')

def estimate_gamma(M, MinvT, lengths, ereff_est, f):
    # obtain an estimate of gamma. This is only used for weighting and solving 
    # for the calibration coefficients. The final gamma is computed from 
    # the calibrated line standards with least-squares.
    
    E = MinvT@M[[0,2,1,3]]
    #options={'rhobeg': 1.0, 'maxiter': 1000, 'disp': False, 'catol': 1e-8}
    xx = so.minimize(findereff, [ereff_est.real, ereff_est.imag],
                     method='COBYLA', #options=options, tol=1e-8,
                   args=(E, lengths, f))
    ereff = xx.x[0] + 1j*xx.x[1]
    gamma = 2*np.pi*f/c0*np.sqrt(-ereff)
    gamma = abs(gamma.real) + 1j*abs(gamma.imag)
    return gamma

def WLS(x,y,w=1, metas=False):
    # Weighted least-squares for a single parameter estimation
    # determine gamma after the calibration coefficients are solved
    dot, inv, eig, solve, \
        conj, exp, log, acosh, sqrt, \
            get_value, ucomplex = metas_or_numpy_funcs(metas=metas)
            
    x = x*(1+0j) # force x to be complex type 
    # return (x.conj().transpose().dot(w).dot(y))/(x.conj().transpose().dot(w).dot(x))
    return dot(x.conj().transpose().dot(w), y)/dot(x.conj().transpose().dot(w), x)

def Vgl(N):
    # inverse covariance matrix for propagation constant computation
    return np.eye(N-1, dtype=complex) - (1/N)*np.ones(shape=(N-1, N-1), dtype=complex)

def compute_gamma(X, M, lengths, gamma_est, metas=False):
    # determine gamma after the calibration coefficients are solved
    dot, inv, eig, solve, \
        conj, exp, log, acosh, sqrt, \
            get_value, ucomplex = metas_or_numpy_funcs(metas=metas)
    
    #EX = solve(X, M)
    EX = dot(inv(X), M)
    gamma_l = log(EX[0,:]/EX[-1,:])
    gamma_l = gamma_l[get_value(lengths) != 0]
    l = -2*lengths[get_value(lengths) != 0]

    n = np.round(get_value(gamma_l - gamma_est*l).imag/np.pi/2)
    gamma_l = gamma_l - 1j*2*np.pi*n # unwrapped
    
    return WLS(l, gamma_l, Vgl(len(l)+1))

def vech(X, upper=True):
    # return vector of the elements in the upper- or lower-triangle of a matrix
    # diagonal elements are excluded!
    N = X.shape[0]
    vechinx = np.triu_indices(N, 1) if upper else np.tril_indices(N, -1)
    return X[vechinx]

def mTRL_at_one_freq(Slines, lengths, Sreflect, ereff_est, reflect_est, f, sw=[0,0]):
    # Performing a standard mTRL without uncertainty. That is, METAS package not used.
    # Slines: array containing 2x2 S-parameters of each line standard
    # lengths: array containing the lengths of the lines
    # Sreflect: 2x2 S-parameters of the measured reflect standard
    # ereff_est: complex scalar of estimated effective permittivity
    # reflect_est: complex scalar of estimated reflection coefficient of the reflect standard
    # f: scalar, current frequency point
    # sw: 1x2 array holding the forward and reverse switch terms, respectively.
    
    # numpy functions
    dot, inv, eig, solve, \
        conj, exp, log, acosh, sqrt, \
            get_value, ucomplex = metas_or_numpy_funcs(metas=False)

    # correct switch term
    Slines = [correct_switch_term(x,sw[0],sw[1]) for x in Slines] if np.any(sw) else Slines
    Sreflect = correct_switch_term(Sreflect,sw[0],sw[1]) if np.any(sw) else Sreflect # this is actually not needed!
    
    # make first line as Thru, i.e., zero length
    lengths = np.array([x-lengths[0] for x in lengths])
    
    # measurements
    Mi    = [S2T(x) for x in Slines] # convert to T-parameters        
    M     = np.array([x.flatten('F') for x in Mi]).T
    MinvT = np.array([inv(x).flatten('F') for x in Mi])
            
    gamma = estimate_gamma(M, MinvT, lengths, ereff_est, f)

    # calibration weighting matrix
    exps = exp(gamma*lengths)
    W = conj(np.outer(exps,1/exps) - np.outer(1/exps,exps))
    
    F = dot(dot(M,W),MinvT[:,[0,2,1,3]]) # weighted measurements
    
    eigval, eigvec = eig(F) # numpy order
    
    # length difference
    dl = np.tile( lengths, (len(lengths), 1) )
    dl = abs(vech(dl) - vech(dl.T))
    
    # sorting the eigenvalues and eigenvectors
    lambd_model = (abs(exp(gamma*dl)-exp(-gamma*dl))**2).sum()
    inx1  = np.argmin(abs(eigval-(-lambd_model)))
    inx2  = np.argmin(abs(eigval-(lambd_model)))
    inx   = [inx1, inx2]
    lambd = (eigval[inx[-1]] - eigval[inx[0]])/2   # lambda for debugging

    x1_ = eigvec[:,inx[0]]
    x1_ = x1_/x1_[0]
    x4  = eigvec[:,inx[-1]]
    x4  = x4/x4[-1]
    x2_ = np.array([x4[2], ucomplex(1), x4[2]*x1_[2], x1_[2]])
    x3_ = np.array([x4[1], x4[1]*x1_[1], ucomplex(1), x1_[1]])
    
    # Normalized calibration matrix
    X_ = np.array([x1_, x2_, x3_, x4]).T
    
    # solve for a11b11 and k from Thru measurement
    ka11b11,_,_,k = solve(X_, M[:,0])
    #ka11b11,_,_,k = dot(inv(X_),M[:,0])
    a11b11  = ka11b11/k
    
    ## solve for a11/b11, a11 and b11
    Ga = Sreflect[0,0]
    Gb = Sreflect[1,1]
    a11_b11 = (Ga - x2_[0])/(1 - Ga*x3_[3])*(1 + Gb*x2_[3])/(Gb + x3_[0])
    a11 = sqrt(a11_b11*a11b11)
    # choose correct answer for a11 and b11
    G0 = get_value( (Ga - x2_[0])/(1 - Ga*x3_[3])/a11 )
    if abs(G0 - reflect_est) > abs(-G0 - reflect_est):
        a11 = -a11
    b11 = a11b11/a11
    reflect_est = get_value( (Ga - x2_[0])/(1 - Ga*x3_[3])/a11 )  # new value
    
    # build the calibration matrix (de-normalize)
    X  = dot(X_,np.diag([a11b11, b11, a11, ucomplex(1)]) )

    gamma = compute_gamma(X, M, lengths, gamma, metas=False)
    ereff = -(c0/2/np.pi/f*gamma)**2
    
    return X, k, get_value(ereff), gamma, get_value(reflect_est), lambd

def cov_ereff_Gamma(ereff_Gamma, lengths, X, k, f):
    # determine cov of ereff_Gamma (line mismatch)
    
    dot, inv, eig, solve, \
        conj, exp, log, acosh, sqrt, \
            get_value, ucomplex = metas_or_numpy_funcs(metas=True)
    
    ereff = ereff_Gamma[0]
    gamma = 2*np.pi*f/c0*sqrt(-ereff)
    G = ereff_Gamma[1]
    Rkron = np.array([[1/(1 - G**2), G/(1 - G**2), -G/(1 - G**2), -G**2/(1 - G**2)],
                      [G/(1 - G**2), 1/(1 - G**2), -G**2/(1 - G**2), -G/(1 - G**2)],
                      [-G/(1 - G**2), -G**2/(1 - G**2), 1/(1 - G**2), G/(1 - G**2)], 
                      [-G**2/(1 - G**2), -G/(1 - G**2), G/(1 - G**2), 1/(1 - G**2)]])
    
    return np.array( [munc.get_covariance( k*X@dot(Rkron, [exp(-gamma*l),0,0,exp(gamma*l)]) ) for l in lengths] )

def umTRL_at_one_freq(Slines, lengths, Sreflect, ereff_est, reflect_est, f, X, k, sw=[0,0],
                      uSlines=None, ulengths=None, uSreflect=None, ureflect=None, uereff_Gamma=None, usw=None):
    # Slines: array containing 2x2 S-paramters of each line standard
    # lengths: array containing the lengths of the lines
    # Sreflect: 2x2 S-paramters of the measured reflect standard
    # ereff_est: complex scalar of estimated effective permittivity
    # reflect_est: complex scalar of estimated reflection coefficient of the reflect standard
    # f: scalar, current frequency point
    # sw: 1x2 array holding the forward and reverse switch terms, respectively.
    # X: 4x4 array estimated calibration coefficients
    # k: scalar of estimated 7th term calibration coefficient
    #
    # uSlines: array containing the 8x8 covariance of each line measurement
    # ulengths: array containing the variance of each line
    # uSreflect: 8x8 covariance matrix of measured reflect 
    # ureflect: 2x2 covariance matrix of the reflection coefficient of the reflect standard
    # uereff_Gamma: 4x4 covariance matrix of the line mismatch.
    # usw: 4x4 covariance matrix of swicth terms
    
    # metas functions
    dot, inv, eig, solve, \
        conj, exp, log, acosh, sqrt, \
            get_value, ucomplex = metas_or_numpy_funcs(metas=True)
    
    if uSlines is not None:
        Slines   = np.array([munc.ucomplexarray(x, covariance=covx) for x,covx in zip(Slines,uSlines)]) 
    else:
        Slines   = np.array([munc.ucomplexarray(x, covariance=np.zeros((8,8))) for x in Slines]) 
    
    if ulengths is not None:
        lengths  = munc.ufloatarray(lengths, covariance=ulengths)
    else:
        lengths  = munc.ufloatarray(lengths, covariance=np.zeros((len(lengths),len(lengths))))
    
    if uSreflect is not None:
        Sreflect = munc.ucomplexarray(Sreflect, covariance=uSreflect)
    else:
        Sreflect = munc.ucomplexarray(Sreflect, covariance=np.zeros((8,8)))
    
    if ureflect is not None:
        reflecta = munc.ucomplex(reflect_est, covariance=ureflect)
        reflectb = munc.ucomplex(reflect_est, covariance=ureflect)
    else:
        reflecta = munc.ucomplex(reflect_est, covariance=np.zeros((2,2)))
        reflectb = munc.ucomplex(reflect_est, covariance=np.zeros((2,2)))
        
    if uereff_Gamma is not None:
        ereff_Gamma = munc.ucomplexarray([ereff_est, 0], covariance=uereff_Gamma)
    else:
        ereff_Gamma = munc.ucomplexarray([ereff_est, 0], covariance=np.zeros((4,4)))
    
    if usw is not None:
        sww = sw # just to use as a check
        sw = munc.ucomplexarray(sw, covariance=usw)
    else:
        sww = sw # just to use as a check
        sw = munc.ucomplexarray(sw, covariance=np.zeros((4,4)))
    
    # correct switch term
    Slines = [correct_switch_term(x,sw[0],sw[1]) for x in Slines] if np.any(sww) else Slines
    Sreflect = correct_switch_term(Sreflect,sw[0],sw[1]) if np.any(sww) else Sreflect # this is actually not needed!
    
    # make first line as Thru, i.e., zero length
    lengths = np.array([x-lengths[0] for x in lengths])
    
    # measurements
    Mi    = [S2T(x) for x in Slines] # convert to T-paramters
    
    # update the covariance of the measurments with impedance mismatch uncertainties
    Mi    = [ munc.ucomplexarray(munc.get_value(x), covariance=munc.get_covariance(x) + cov)
             for x,cov in zip(Mi, cov_ereff_Gamma(ereff_Gamma, munc.get_value(lengths), X, k, f))]
    
    M     = np.array([x.flatten('F') for x in Mi]).T
    MinvT = np.array([inv(x).flatten('F') for x in Mi])
    
    gamma = estimate_gamma(get_value(M), get_value(MinvT), get_value(lengths), ereff_est, f)
    
    # calibration weighting matrix
    exps = exp(gamma*lengths)
    W = conj(np.outer(exps,1/exps) - np.outer(1/exps,exps))
    
    F = dot(dot(M,W),MinvT[:,[0,2,1,3]]) # weighted measurements
    
    eigvec, eigval = eig(F) # metas order
    
    dl = np.tile( lengths, (len(lengths), 1) )
    dl = abs(vech(dl) - vech(dl.T))
    lambd_model = (abs(exp(gamma*dl)-exp(-gamma*dl))**2).sum()
    inx1  = np.argmin( abs( get_value(eigval-(-lambd_model)) ) )
    inx2  = np.argmin( abs( get_value(eigval-(lambd_model)) ) )
    inx   = [inx1, inx2]
    lambd = (eigval[inx[-1]] - eigval[inx[0]])/2   # lambda. For debugging
    
    x1_ = eigvec[:,inx[0]]
    x1_ = x1_/x1_[0]
    x4  = eigvec[:,inx[-1]]
    x4  = x4/x4[-1]
    x2_ = np.array([x4[2], ucomplex(1), x4[2]*x1_[2], x1_[2]])
    x3_ = np.array([x4[1], x4[1]*x1_[1], ucomplex(1), x1_[1]])
    # Normalized calibration matrix 
    X_ = np.array([x1_, x2_, x3_, x4]).T
    
    # solve for a11b11 and k from Thru measurement
    ka11b11,_,_,k = solve(X_, M[:,0])
    #ka11b11,_,_,k = dot(inv(X_),M[:,0])
    a11b11  = ka11b11/k
    
    ## solve for a11/b11, a11 and b11
    Ga = Sreflect[0,0]
    Gb = Sreflect[1,1]
    a11_b11 = (reflectb/reflecta)*(Ga - x2_[0])/(1 - Ga*x3_[3])*(1 + Gb*x2_[3])/(Gb + x3_[0])
    a11 = sqrt(a11_b11*a11b11)
    # choose correct answer for a11 and b11
    G0 = get_value( (Ga - x2_[0])/(1 - Ga*x3_[3])/a11 )
    if abs(G0 - reflect_est) > abs(-G0 - reflect_est):
        a11 = -a11
    b11 = a11b11/a11
    reflect_est = get_value( (Ga - x2_[0])/(1 - Ga*x3_[3])/a11 )  # new value
    
    # build the calibration matrix (de-normalize)
    X  = dot(X_,np.diag([a11b11, b11, a11, ucomplex(1)]) )
    
    gamma = compute_gamma(X, M, lengths, gamma, metas=True)
    ereff = -(c0/2/np.pi/f*gamma)**2
    
    return X, k, get_value(ereff), gamma, get_value(reflect_est), lambd

def convert2cov(x, f_length, cov_length=2):
    '''
    make input into covariance matrix
    f_length is the number of frequeny points
    cov_length is the final diagonal length of the cov matrix
    
    Three cases are considerd:
        1. the input is a scalar variance --> convert to diagonal with same variance --> repeat along frequency axes
        2. the input is a vector variance --> only diagonalize it --> repeat along frequency axes
        3. the input is 2D matrix --> do nothing --> repeat along frequency axes
        4. the input is 3D array --> do nothing --> do nothing (assuming the user knows what he/she is doing!)
    '''
    f_length   = int(f_length)
    cov_length = int(cov_length)
    x = np.atleast_1d(x)
    
    if len(x.shape) > 1:
        if len(x.shape) > 2:
            cov = x  # assume everything is fine
        else:
            cov = np.tile(x, (f_length,1,1))
    else:
        if x.shape[0] > 1:
            cov = np.tile(np.diag(x), (f_length,1,1))
        else:
            cov = np.tile(np.eye(cov_length)*x[0], (f_length,1,1))
    
    return cov

class umTRL:
    """
    Multiline TRL calibration with uncertainty capabilities, hence "u"mTRL.
    
    This is an extension of my original mTRL algorithm [1] to account 
    for uncertainties with the help of METAS UncLib [2].

    """
    
    def __init__(self, lines, line_lengths, reflect, 
                 reflect_est=-1, reflect_offset=0, ereff_est=1+0j, switch_term=None,
                 uSlines=None, ulengths=None, uSreflect=None, ureflect=None, uereff_Gamma=None, uswitch_term=None):
        """
        umTRL initializer.
        """

        self.f  = lines[0].frequency.f
        self.Slines = np.array([x.s for x in lines])
        self.lengths = np.array(line_lengths)
        self.Sreflect = reflect.s
        self.reflect_est = reflect_est
        self.reflect_offset = reflect_offset
        self.ereff_est = ereff_est
        
        if switch_term is not None:
            self.switch_term = np.array([x.s.squeeze() for x in switch_term])
        else:
            self.switch_term = np.array([self.f*0 for x in range(2)])
        
        # uncertainties
        self.uSlines      = np.array(uSlines) if uSlines is not None else np.zeros(len(self.lengths))
        if len(self.uSlines.shape) < 3:
            # handle cases when user only give one variance/covaraince for all measurements
            if len(self.uSlines.shape) < 2:
                if len(self.uSlines.shape) < 1:
                    self.uSlines  = np.array([self.uSlines.squeeze() for l in self.lengths])
            else:
                self.uSlines  = np.array([self.uSlines.squeeze() for l in self.lengths])
            
        self.ulengths     = ulengths if ulengths is not None else 0
        self.uSreflect    = uSreflect if uSreflect is not None else 0
        self.ureflect     = ureflect if ureflect is not None else 0
        self.uereff_Gamma = uereff_Gamma if uereff_Gamma is not None else 0
        self.usw          = uswitch_term if uswitch_term is not None else 0
        
    
    def run_mTRL(self):
        # This runs the standard mTRL without uncertainties (very fast).
        gammas  = []
        lambds   = []
        Xs      = []
        ks      = []
        ereff0  = self.ereff_est
        gamma0  = 2*np.pi*self.f[0]/c0*np.sqrt(-ereff0)
        reflect_est0 = -1*np.exp(-2*gamma0*self.reflect_offset)
        
        lengths = self.lengths

        for inx, f in enumerate(self.f):
            Slines = self.Slines[:,inx,:,:]
            Sreflect = self.Sreflect[inx,:,:]
            sw = self.switch_term[:,inx]
            
            X,k,ereff0,gamma,reflect_est0,lambd = mTRL_at_one_freq(Slines, lengths, Sreflect, 
                                                    ereff_est=ereff0, reflect_est=reflect_est0, 
                                                    f=f, sw=sw)
            Xs.append(X)
            ks.append(k)
            gammas.append(gamma)
            lambds.append(lambd)
            print(f'Frequency: {f*1e-9:0.2f} GHz')

        self.X = np.array(Xs)
        self.k = np.array(ks)
        self.gamma = np.array(gammas)
        self.ereff = -(c0/2/np.pi/self.f*self.gamma)**2
        self.lambd = np.array(lambds)
        self.error_coef()
        
    def run_umTRL(self):
        # This runs the mTRL with uncertainties 
        # (quite slow because of METAS UncLib, but faster than a MC analysis ;) ).
        
        self.run_mTRL() # initial run to get nominal cal coefficients
        
        gammas  = []
        lambds  = []
        Xs      = []
        ks      = []
        ereff0  = self.ereff[0]
        gamma0  = 2*np.pi*self.f[0]/c0*np.sqrt(-ereff0)
        reflect_est0 = -1*np.exp(-2*gamma0*self.reflect_offset)
        
        # line lengths
        lengths = self.lengths
        
        uSlines_full      = np.array([ convert2cov(x, len(self.f), 8) for x in self.uSlines ])
        ulengths_full     = convert2cov(self.ulengths, len(self.f), len(lengths))
        uSreflect_full    = convert2cov(self.uSreflect, len(self.f), 8)
        ureflect_full     = convert2cov(self.ureflect, len(self.f), 2)
        uereff_Gamma_full = convert2cov(self.uereff_Gamma, len(self.f), 4)
        usw_full          = convert2cov(self.usw, len(self.f), 4)
        
        for inx, f in enumerate(self.f):
            Slines = self.Slines[:,inx,:,:]
            Sreflect = self.Sreflect[inx,:,:]
            sw = self.switch_term[:,inx]
            
            # uncertainties
            uSlines   = uSlines_full[:,inx,:,:]
            ulengths  = ulengths_full[inx,:,:]
            uSreflect = uSreflect_full[inx,:,:]
            ureflect  = ureflect_full[inx,:,:]
            uereff_Gamma = uereff_Gamma_full[inx,:,:]
            usw   = usw_full[inx,:,:]
            
            X,k,ereff0,gamma,reflect_est0,lambd = \
                umTRL_at_one_freq(Slines, lengths, Sreflect, 
                                  ereff_est=ereff0, reflect_est=reflect_est0, f=f,
                                  X=self.X[inx], k=self.k[inx], sw=sw,
                                  uSlines=uSlines, ulengths=ulengths, 
                                  uSreflect=uSreflect, ureflect=ureflect,
                                  uereff_Gamma=uereff_Gamma, usw=usw
                                  )
            Xs.append(X)
            ks.append(k)
            gammas.append(gamma)
            lambds.append(lambd)
            print(f'Frequency: {f*1e-9:0.2f} GHz')

        self.X = np.array(Xs)
        self.k = np.array(ks)
        self.gamma = np.array(gammas)
        self.ereff = -(c0/2/np.pi/self.f*self.gamma)**2
        self.lambd = np.array(lambds)
        self.error_coef()
    
    def error_coef(self):
        # return the 3 error terms of each port
        #
        # R. B. Marks, "Formulations of the Basic Vector Network Analyzer Error Model including Switch-Terms," 
        # 50th ARFTG Conference Digest, 1997, pp. 115-126.
        #
        # left port:
        # ERF: forward reflection tracking
        # EDF: forward directivity
        # ESF: forward source match
        # 
        # right port:
        # ERR: reverse reflection tracking
        # EDR: reverse directivity
        # ESR: reverse source match
        
        X = self.X
        self.coefs = {}
        
        # forward errors
        self.coefs['ERF'] =  X[:,2,2] - X[:,2,3]*X[:,3,2]
        self.coefs['EDF'] =  X[:,2,3]
        self.coefs['ESF'] = -X[:,3,2]
        
        # reverse errors
        self.coefs['ERR'] =  X[:,1,1] - X[:,3,1]*X[:,1,3]
        self.coefs['EDR'] = -X[:,1,3]
        self.coefs['ESR'] =  X[:,3,1]
        
        return self.coefs
    
    def apply_cal(self, NW, left=True):
        # apply calibration to a 1-port or 2-port network.
        # NW:   the network to be calibrated (1- or 2-port).
        # left: boolean: define which port to use when 1-port network is given.
        # If left is True, left port is used; otherwise right port is used.
        # The outputs are the corrected NW and its cov matrix (at every freq.)
        
        nports = np.sqrt(len(NW.port_tuples)).astype('int') # number of ports
        # if 1-port, convert to 2-port (later convert back to 1-port)
        if nports < 2:
            NW = rf.two_port_reflect(NW)
        
        metas = True if isinstance(self.k[0], type(munc.ucomplex(0))) else False
        
        # numpy or metas functions
        dot, inv, eig, solve, \
            conj, exp, log, acosh, sqrt, \
                get_value, ucomplex = metas_or_numpy_funcs(metas=metas)
        
        # apply cal
        S_cal = []
        cov   = []
        for x,k,s,sw in zip(self.X, self.k, NW.s, self.switch_term.T):
            s    = correct_switch_term(s, sw[0], sw[1]) if np.any(sw) else s
            xinv = inv(x)
            s11 = ucomplex(complex(s[0,0]))
            s21 = ucomplex(complex(s[1,0]))
            s12 = ucomplex(complex(s[0,1]))
            s22 = ucomplex(complex(s[1,1]))
            M_ = np.array([-s11*s22+s12*s21, -s22, s11, 1])
            T_ = dot(xinv, M_)
            s21_cal = k*s21/T_[-1]
            T_ = T_/T_[-1]
            scal = np.array([[T_[2], (T_[0]-T_[2]*T_[1])/s21_cal],[s21_cal, -T_[1]]])
            if metas:
                S_cal.append(munc.get_value(scal))
                cov.append(munc.get_covariance(scal))
            else:
                S_cal.append(scal)
                cov.append(np.zeros((scal.size*2, scal.size*2)))
            
        S_cal = np.array(S_cal)
        cov   = np.array(cov)
                
        # revert to 1-port device if the input was a 1-port device
        if nports < 2:
            if left: # left port
                S_cal = S_cal[:,0,0]
            else:  # right port
                S_cal = S_cal[:,1,1]
        
        return rf.Network(frequency=NW.frequency, s=S_cal.squeeze()), cov
    
    def shift_plane(self, d=0):
        # shift calibration plane by distance d
        # negative: shift toward port
        # positive: shift away from port
        # e.g., if your Thru has a length of L, 
        # then d=-L/2 to shift the plane backward 
        X_new = []
        K_new = []
        for x,k,g in zip(self.X, self.k, self.gamma):
            z = np.exp(-g*d)
            KX_new = k*x.dot(np.diag([z**2, 1, 1, 1/z**2]))
            X_new.append(KX_new/KX_new[-1,-1])
            K_new.append(KX_new[-1,-1])
            
        self.X = np.array(X_new)
        self.K = np.array(K_new)
    
    def renorm_impedance(self, Z_new, Z0=50):
        # re-normalize reference calibration impedance
        # by default, the ref impedance is the characteristic 
        # impedance of the line standards.
        # Z_new: new ref. impedance (can be array if frequency dependent)
        # Z0: old ref. impedance (can be array if frequency dependent)
        
        # ensure correct array dimensions (if not, you get an error!)
        N = len(self.k)
        Z_new = Z_new*np.ones(N)
        Z0    = Z0*np.ones(N)
        
        G = (Z_new-Z0)/(Z_new+Z0)
        X_new = []
        K_new = []
        for x,k,g in zip(self.X, self.k, G):
            KX_new = k*x.dot( np.kron([[1, -g],[-g, 1]],[[1, g],[g, 1]])/(1-g**2) )
            X_new.append(KX_new/KX_new[-1,-1])
            K_new.append(KX_new[-1,-1])

        self.X = np.array(X_new)
        self.K = np.array(K_new)